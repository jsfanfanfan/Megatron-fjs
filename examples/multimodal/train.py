import os
import sys
from functools import partial

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

from dataloader_provider import train_valid_test_dataloaders_provider
from model import model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import is_last_rank

def get_batch(data_iterator):
    """
    输入一个数据迭代器，
    返回一个batch的数据
    """

    args = get_args()

    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None
    num_tiles = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(["text"], data, torch.int64)["text"]
    prompt_len = tensor_parallel.broadcast_data(["prompt_len"], data, torch.int64)["prompt_len"]
    target = tensor_parallel.broadcast_data(["target"], data, torch.int64)["target"]

    imgs = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
    num_tiles = tensor_parallel.broadcast_data(["num_tiles"], data, torch.int)["num_tiles"]

    # Dummy image, no image.
    if imgs.shape == torch.Size([1, 1]):
        imgs = torch.tensor([], dtype=torch.float32, device=data_text.device)
        num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)

    torch.cuda.nvtx.range_pop()

    tokens_ = data_text.long()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()
    text_length = tokens_.shape[1]
    tokens = tokens_[:, :text_length].contiguous()
    labels = target[:, 1:text_length+1].contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    attention_mask, loss_mask, position_ids = \
        get_ltor_masks_and_position_ids(tokens, eod_token,
                                        args.reset_position_ids,
                                        args.reset_attention_mask,
                                        args.eod_mask_loss,
                                        question_length=prompt_len,
                                        target=target[:, 1:text_length+1]
                                        )
    torch.cuda.nvtx.range_pop()

    return tokens, labels, loss_mask, attention_mask, position_ids, imgs, num_tiles


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    question_length=None,
                                    target=None,
                                    weights=None):
    """建立 masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

     # Loss mask.
    if target != None: # use target to create loss mask that is created in data preparation step
        loss_mask = torch.ones(target.size(), dtype=torch.float, device=data.device)
        loss_mask[target == eod_token] = 0.0 # mask paddings
        loss_mask[target == -100] = 0.0 # mask prompts

    else: # default creation
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
        if eod_mask_loss:
            loss_mask[data == eod_token] = 0.0

        if question_length is not None:
            # Create a mask based on question_length
            question_length_mask = torch.arange(loss_mask.size(1), device=loss_mask.device)[None, :] < question_length[:, None]
            # Invert the mask (1 where we want to keep the loss, 0 where we want to zero it out)
            inverted_mask = ~question_length_mask
            # Apply the mask to loss_mask
            loss_mask = loss_mask * inverted_mask.float()

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)
    if weights is not None:
        loss_mask = loss_mask * weights

    return attention_mask, loss_mask, position_ids


def loss_func(loss_mask, output_tensor):
    """
    输入损失mask和输出张量
    返回损失和local_num_token
    """
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum()
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    loss = torch.cat([total_loss.view(1), total_tokens.view(1)])

    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    return (
        total_loss,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: LLaVAModel):
    """
    输入数据迭代器和模型
    返回输出张量和特定 loss mask 的 loss_func
    """
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images, num_image_titles = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(images, tokens, position_ids, attention_mask, labels, loss_mask, num_image_titles=num_image_titles)

    return output_tensor, partial(loss_func, loss_mask)


def llava_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank: # 感觉这里越界，其实没有，用的 encoder
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]
    

def llava_position_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]
    

def run_online_eval(model):
    """在训练时跑一个 evaluation benchmark."""
    args = get_args()

    # Online evaluation config is not defined. Do nothing.
    if not args.online_evaluation_config:
        return []

    from config import EvaluationConfig
    from run_text_generation import generate_and_write_samples, patch_tokenizer

    with open(args.online_evaluation_config, "r") as f:
        config_dict = yaml.safe_load(f)

    config = EvaluationConfig(**config_dict)

    patch_tokenizer(args)

    # The inference code assumes the first rank is the leader.
    # Tensorboard writer is on the last rank.
    # We must write to a storage space that all ranks see.
    output_dir = os.path.join(args.save, "online_eval")
    os.makedirs(output_dir, exist_ok=True)
    config.output_path = os.path.join(output_dir, args.language_model_type)

    # The actual generation.
    generate_and_write_samples(model[0].module, config, print_output=False)

    # Make sure the first rank is done writing so that the last rank can run eval.
    torch.distributed.barrier()

    if not is_last_rank():
        return []

    # Run evaluation.
    if config.task == "TextVQA":
        from evaluate_textvqa import textvqa_eval
        avg_acc = textvqa_eval(config.output_path)

        return [{"TextVQA accuracy": avg_acc}]
    else:
        raise NotImplementedError(f"online evaluation of {config.task} not implemented yet")
    

def write_online_eval_to_tensorboard(data, iteration, writer):
    """把在线评估结果写到 Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider, # 返回一个 EnergonDataloader
        model_provider, # 返回一个 LLava model
        ModelType.encoder_and_decoder,
        forward_step, # 前向一个 batch
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_multimodal_extra_args, # multimodel_args.py
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
        non_loss_data_func=run_online_eval
    )