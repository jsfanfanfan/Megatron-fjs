import dataclasses
from datetime import datetime
import functools
import gc
import logging
import math
import os
import sys
from .log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
    is_float8tensor,
)
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.training.checkpointing import checkpoint_exists
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.parallel_state import (
    destroy_global_memory_buffer,
    destroy_model_parallel,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)

from .async_utils import maybe_finalize_async_save
from .utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    append_to_progress_log,
    update_use_dist_ckpt,
)
from .global_vars import (
    destroy_global_vars,
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger)
from . import one_logger_utils

from . import ft_integration




# 使用cuda()事件对选定的操作进行计时
stimier = StragglerDetector()

def destroy_global_state():
    destroy_global_vars()
    destroy_num_microbatches_calculator()
    destroy_global_memory_buffer()
    destroy_model_parallel()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


# 通过batchsize计算浮点操作的数量
def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )

    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2

    return (
        expansion_factor
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    + (args.seq_length / args.hidden_size)
                ) * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Shared Experts.
            + ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )


def get_start_time_from_progress_log():
    """
    获取具有相同 world size 的最早作业的开始时间。
    同时返回上次保存的检查点中完成的浮点运算次数。
    """
    args = get_args()
    assert args.save is not None
    progress_log_filename = os.path.join(args.save, "progress.txt")

    # start_time 是具有相同 world size 的作业开始的时间。
    # start_num_floating_point_operations 
    # 是该作业开始时已经完成的浮点运算次数。
    # latest_num_floating_point_operations 
    # 是最近保存的检查点中已完成的浮点运算次数。
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(': ')[1])

    with open(progress_log_filename, 'r') as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split('\t')
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = \
                    _get_field(line_tokens[7], float)
            if world_size_in_line != args.world_size:
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = \
                        latest_num_floating_point_operations
    assert start_time is not None and start_num_floating_point_operations is not None, \
        "Should have seen at least one 'Starting job' entry with same world_size"
    return datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), \
        start_num_floating_point_operations


def pretrain(
        train_valid_test_dataset_provier,
        model_provider,
        model_type,
        forward_step_func,
        process_non_loss_data_func=None,
        extra_args_provider=None,
        args_default={},
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        non_loss_data_func=None,
):
    """主训练程序
    
    这个函数将按照以下顺序执行
        1）初始化Megatron
        2）使用 model_provider 建立模型，优化器，学习率调度器
        3）调用 train_valid_test_dataset_provier 获取数据集
        4）使用 forward_step_func 训练模型
        
        参数：
            train_valid_test_dataset_provider: 一个函数，它接受训练、验证和测试数据集的大小，
                并返回 train、valid、test 数据集。
            model_provider: 一个函数，它返回模型的基础版本。
                这里的基础版本指的是一个简单的 CPU 模型，没有使用 FP16（半精度）或 DDP（分布式数据并行）。
            model_type: 一个枚举值，用于指定正在训练的模型的类型。
            forward_step_func: 一个函数，它接受 数据迭代器 和 模型，
                并返回一个包含 损失 标量和字典的结果，其中字典的键值对是训练过程中
                需要监控的信息，例如 lm-loss: value。
                我们还要求这个函数将 批次生成器 添加到计时器类中。

            process_non_loss_data_func: 一个函数，用于后处理网络的输出。
                它可以用于将输出张量（如图像）写入 TensorBoard。
                该函数接受 收集的数据（张量列表）、当前迭代索引 和 TensorBoard writer 作为参数。

            extra_args_provider: 一个函数，接受一个解析器（parser）并向其中添加参数。
                用于为程序添加自定义参数。
            args_defaults: 一个字典，用于将参数名称映射到默认值。
                用于设置已经解析过的参数的默认值。
            get_embedding_ranks (TODO):
            get_position_embedding_ranks (TODO):
            non_loss_data_func (callable): 在 evaluation 过程中调用的函数.
                It can run e.g. benchmarks
        """
    
    # 初始化并且获取参数，计时器和 Tensorboard writer
    # 建立 logging
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_default=args_default,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    args = get_args()
    timers = get_timers()

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == 'local':
        raise RuntimeError('LocalCheckpointManagers are not yet integrated')
        checkpointing_context = {
            'local_checkpoint_manager': BasicLocalCheckpointManager(
                args.non_persistent_local_ckpt_dir
            )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, checkpointing_context=checkpointing_context)
    
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()



def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):

    return model


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              checkpointing_context=None):
    """Set up model and optimizer"""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    model = get_model(model_provider_func, model_type)