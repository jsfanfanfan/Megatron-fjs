"""Utility functions used throughout Megatron core"""
import array
import hashlib
import logging
import math
import operator
import queue
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from importlib.metadata import version
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from packaging.version import Version as PkgVersion

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor









def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, '__version__'):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    global _te_version
    if _te_version is None:
        _te_version = PkgVersion(get_te_version_str())
    return _te_version


def is_te_min_version(version, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    if check_equality:
        return get_te_version() >= PkgVersion(version)
    return get_te_version() > PkgVersion(version)


class StragglerDetector:
    """Singleton Class implementing per rank Straggler Detector

    它使用 CUDA 事件（cuda events）来对选定的操作进行计时，
    使用 start 和 stop 方法，这些方法可以直接通过类实例调用，
    也可以像 Python 上下文管理器一样使用。在收集完数据后，
    可以使用 report() 方法显示收集到的指标。
    此功能仅在 CUDA 可用的情况下支持。
    更多信息请参见 megatron/core/README_STRAGGLER.md

    Note:
        The instance and class attributes mentioned below are all
        private to the class and has no use outside the class

    Attributes:
        _off (bool): current state of the toggle
        start (FunctionType): start method
        stop (FunctionType): stop method
        world (int): world size
        rank (int): rank for this instance
        mmcnt (int): number of ranks to report
        port (int): control port
        amp (float): amplification factor for TFLOPs, default 3.0
        toggle (bool): whether to start/stop detector collection
        bdata (bool): when true, just collect get_batch
        dev (int): cuda device
        evt_q (LifoQueue): cuda event queue
        start_gemm_ev (list[torch.cuda.Event]): cuda start event
        stop_gemm_ev (list[torch.cuda.Event]): cuda stop event
        start_data_ev (list[torch.cuda.Event]): cuda start event
        stop_data_ev (list[torch.cuda.Event]): cuda stop event
        start_gemm_tm (list[int]): start time (wallclock)
        stop_gemm_tm (list[int]): stop time (wallclock)
        start_data_tm (list[int]): start time for get_batch
        stop_data_tm (list[int]): stop time for get_batch
        sock (socket): the controller socket
        ctrlr (Thread): the controller thread
    """

    _configured = False
    """Indicates if the singleton instance is configured or not
    """

    def __new__(cls: Type["StragglerDetector"]) -> "StragglerDetector":
        """Constructor
        Creates an instance of the class if not created

        Args:
            cls (Type[&#39;StragglerDetector&#39;]): The class type

        Returns:
            StragglerDetector: the class instance
        """

        if not hasattr(cls, "_instance"):
            cls._instance = super(StragglerDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initializer

        The inital state of the StragglerDetector instance is disabled.
        The enabled state is indicated using self._off member variable
        and the proerty enabled.
        """
        self._off: bool = True
        self.start = self.null_method
        self.stop = self.null_method
        self.world: int = 0
        self.rank: int = 0
        self.mmcnt: int = 1
        self.port: int = 0
        self.amp: float = 3.0
        self.toggle: bool = False
        self.bdata: bool = False
        self.dev: Union[torch.device, int, None] = None
        self.evt_q: Union[queue.LifoQueue, None] = None
        self.start_gemm_ev: List[torch.cuda.Event] = []
        self.stop_gemm_ev: List[torch.cuda.Event] = []
        self.start_data_ev: List[torch.cuda.Event] = []
        self.stop_data_ev: List[torch.cuda.Event] = []
        self.start_gemm_tm: List[int] = []
        self.stop_gemm_tm: List[int] = []
        self.start_data_tm: List[int] = []
        self.stop_data_tm: List[int] = []
        self.sock: Union[socket.socket, None] = None
        self.ctrlr: Union[threading.Thread, None] = None

    def configure(
        self,
        world: int,
        rank: int,
        mmcnt: int = 1,
        amp: float = 3.0,
        port: int = 65535,
        prefill: int = 1024,
        enabled: bool = False,
    ) -> None:
        """This method is called to configure the Singleton instance

        It should be called once per instantiation per process.

        Note:
            The constructor keeps the state of instance disabled
            i.e no collection will happen even when start/stop methods are
            called. Only when enabled is True (self._off is True), the
            start/stop method pointers get assigned the real collection
            methods, otherwise they are initialized with null_method

        Args:
            world (int): World Size
            rank (int): The rank of this trainer
            mmcnt (int, optional): Number of ranks to print for showing Min/Max Etpt.
                                   Defaults to 1.
            amp (float, optional): Set to 3.0 if we only use timers in fwd pass.
                                   Defaults to 3.0.
            port (int, optional): Control port, useful only for rank-0. Defaults to 65535.
            prefill (int, optional): Howmany Events to pre-populate. Defaults to 1024.
            enabled (bool, optional): Whether or not collection is enabled on startup.
                                      Defaults to False.
        """
        if StragglerDetector._configured:
            # don't throw
            return
        StragglerDetector._configured = True
        self.bdata = False
        self.start = self.null_method
        self.stop = self.null_method
        self._off = True
        # No CUDA, No Support
        if torch.cuda.is_available():
            self._off = not enabled
            self.world = world
            self.rank = rank
            self.mmcnt = mmcnt if mmcnt > 1 else 1
            self.amp = amp
            self.port = port
            self.toggle = False
            self.bdata = False
            self.evt_q = queue.LifoQueue()
            self.start_gemm_ev = []
            self.stop_gemm_ev = []
            self.start_data_ev = []
            self.stop_data_ev = []
            self.start_gemm_tm = []
            self.stop_gemm_tm = []
            self.start_data_tm = []
            self.stop_data_tm = []
            backend = torch.distributed.get_backend()
            if backend == "nccl":
                self.dev = torch.cuda.current_device()
            else:
                self.dev = torch.device("cpu")
            # cache some events
            for _ in range(prefill):
                self.evt_q.put(torch.cuda.Event(enable_timing=True))
            if self.rank == 0:
                # Start the controller
                self._controller()
            if not self._off:
                self.start = self.start_method
                self.stop = self.stop_method

    def reset(self) -> None:
        """This method is called to reset the metrics state of the instance

        It is generally called from within elapsed() after extracting per rank metrics.
        """
        if self._off:
            return
        # Pool them
        if self.evt_q is not None:
            _ = [self.evt_q.put(ev) for ev in self.start_gemm_ev]
            _ = [self.evt_q.put(ev) for ev in self.stop_gemm_ev]
            _ = [self.evt_q.put(ev) for ev in self.start_data_ev]
            _ = [self.evt_q.put(ev) for ev in self.stop_data_ev]
        self.start_gemm_ev = []
        self.stop_gemm_ev = []
        self.start_data_ev = []
        self.stop_data_ev = []
        # Use regular timers
        self.start_gemm_tm = []
        self.stop_gemm_tm = []
        self.start_data_tm = []
        self.stop_data_tm = []
        self.bdata = False

    def start_method(self) -> None:
        """This method adds the start timers.

        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. This way it can be used to measure time spent on
        CPU - generally useful for timing get_batch()
        """
        # Not reentrant
        if self.evt_q is not None and self.evt_q.qsize() > 1:
            sev = self.evt_q.get()  # no try-catch
            eev = self.evt_q.get()  # no try-catch
        else:
            sev = torch.cuda.Event(enable_timing=True)
            eev = torch.cuda.Event(enable_timing=True)
        # First check if this start is for data
        if self.bdata:
            self.start_data_ev.append(sev)
            self.stop_data_ev.append(eev)
            self.start_data_tm.append(0)
            self.stop_data_tm.append(0)
            idx = len(self.stop_data_tm) - 1
            self.start_data_tm[idx] = time.perf_counter_ns()
            self.start_data_ev[idx].record()
            self.bdata = False
            return
        self.start_gemm_ev.append(sev)
        self.stop_gemm_ev.append(eev)
        self.start_gemm_tm.append(0)
        self.stop_gemm_tm.append(0)
        idx = len(self.stop_gemm_tm) - 1
        self.start_gemm_tm[idx] = time.perf_counter_ns()
        self.start_gemm_ev[idx].record()

    def stop_method(self) -> None:
        """This method adds the stop timers.

        Both cuda event and perf_counter are added. If bdata is set to
        true from __call__, this method skips inserting cuda
        timer. Also see start_method()
        """
        # Not reentrant
        # First check if this stop is for data
        idx = len(self.stop_data_tm) - 1
        if idx >= 0 and self.stop_data_tm[idx] == 0:
            self.stop_data_tm[idx] = time.perf_counter_ns()
            self.stop_data_ev[idx].record()
            return
        idx = len(self.stop_gemm_tm) - 1
        if idx >= 0 and self.stop_gemm_tm[idx] == 0:
            self.stop_gemm_tm[idx] = time.perf_counter_ns()
            self.stop_gemm_ev[idx].record()

    def elapsed(self) -> Tuple[float, float, int, int, int, int]:
        """This method is called from report(), or can be called directly

         It is called to collect all the elapsed time since last reset().
         It finally calls reset()

        Returns:
            Tuple[float, float, int, int, int, int]: see below for returns
                delta       : time spent in kernel
                batch_delta : time spent in get_batch
                temp        : observed gpu temp
                power       : observed gpu power
                util        : observed gpu utilization
                clock       : observed gpu clock
        """
        if self._off:
            # match with return below
            return 0, 0, 0, 0, 0, 0
        ls_ev = len(self.start_gemm_ev)
        le_ev = len(self.stop_gemm_ev)
        ls_bs = len(self.start_data_ev)
        ls_be = len(self.stop_data_ev)
        delta = 0.0
        batch_delta = 0.0
        temp = 0
        power = 0
        clock = 0
        if ls_ev != le_ev:
            logger.warning(f"Event Start/Stop out of sync {ls_ev}/{le_ev}")
        elif ls_bs != ls_be:
            logger.warning(f"get_batch Start/Stop out of sync {ls_bs}/{ls_be}")
        else:
            temp = torch.cuda.temperature()
            power = torch.cuda.power_draw()
            util = torch.cuda.utilization()
            clock = torch.cuda.clock_rate()
            torch.cuda.synchronize()
            # Process Events
            for i in range(ls_ev):
                e_ev = self.start_gemm_ev[i].elapsed_time(self.stop_gemm_ev[i])
                e_tm = (self.stop_gemm_tm[i] - self.start_gemm_tm[i]) / 1e6  # ns to ms
                # Pick the larger of Event and perf_counter time?
                delta += max(e_ev, e_tm)
            # Process get_batch
            for i in range(ls_bs):
                b_ev = self.start_data_ev[i].elapsed_time(self.stop_data_ev[i])
                b_tm = (self.stop_data_tm[i] - self.start_data_tm[i]) / 1e6  # ns to ms
                # data fetching has prefetch, hence take the max, instead of avg
                batch_delta = max(batch_delta, max(b_ev, b_tm))
        self.reset()  # Prepare for next round
        # time in ms, batch_delta in ms, check return above
        return delta, batch_delta, temp, power, util, clock

    def report(self, total_flops: float = 0.0, log_interval: int = 0) -> bool:
        """Function to log the min/max metircs and the associated rank over a time period

        It finds the slowest and fastest rank among all ranks. It should be
        called by all ranks, but only rank-0 prints the analysis
        At the end it checks, if the straggler detector should
        remain active or if it should be deactivated.

        Args:
            total_flops (float, optional): The theoretical flops over the period. Defaults to 0.0.
            log_interval (int, optional): The training interval over which reporting is called(ms)
                                          Defaults to 0.

        Returns:
            bool: True if reported, else False
        """
        ret = False
        if not self._off and total_flops > 0.0 and log_interval > 0:
            elapsed, btime, temp, power, util, clock = self.elapsed()  # get raw time
            # btime (get_batch time is max in the iteration)
            ptime = elapsed / (log_interval * 1.0)  # avg per iteration elapsed time, ms
            api_flops = total_flops / (log_interval * 1.0)  # avg per iteration flops, ms
            apir_flops = api_flops / (
                ptime * 10**9 * self.world
            )  # this is avg per iteration this rank's thruput, TFLOP/s (note 10**9),
            et_flops = apir_flops / self.amp  # Estimated TFLOPs, not tracing backward

            o_dt = self._min_max(
                ptime, btime, float(temp), float(power), float(util), float(clock), et_flops
            )
            if self.rank == 0 and o_dt is not None and o_dt.aflops is not None:
                now = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                min_flops, min_frank, _ = o_dt.aflops[0]()
                max_flops, max_frank, _ = o_dt.aflops[-1]()
                logger.info(
                    f"{now} | "
                    f"MnRtt/Rnk: {o_dt.min_elapsed} | "
                    f"MxRtt/Rnk: {o_dt.max_elapsed} | "
                    f"MnPwr/Rnk: {o_dt.min_power} | "
                    f"MxPwr/Rnk: {o_dt.max_power} | "
                    f"MnTmp/Rnk: {o_dt.min_temp} | "
                    f"MxTmp/Rnk: {o_dt.max_temp} | "
                    f"MnUtl/Rnk: {o_dt.min_util} | "
                    f"MxUtl/Rnk: {o_dt.max_util} | "
                    f"MnClk/Rnk: {o_dt.min_clock} | "
                    f"MxClk/Rnk: {o_dt.max_clock} | "
                    f"MnDRtt/Rnk: {o_dt.min_btime} | "
                    f"MxDRtt/Rnk: {o_dt.max_btime} | "
                    f"MnEtpt/Rnk: {min_flops:.2f}TF/{min_frank} | "
                    f"MxEtpt/Rnk: {max_flops:.2f}TF/{max_frank}"
                )
                if self.mmcnt > 1 and self.mmcnt < self.world:
                    line = f"^^^^ Bottom {self.mmcnt} Ranks with lowest  Etpt(TF):"
                    for i in range(self.mmcnt):
                        line += f" {o_dt.aflops[i]},"
                    logger.info(line)
                    line = f"^^^^ Top    {self.mmcnt} Ranks with highest Etpt(TF):"
                    shift = self.world - self.mmcnt
                    for i in range(self.mmcnt):
                        line += f" {o_dt.aflops[i+shift]},"
                    logger.info(line)
                ret = True

        # Check/Communicate if tracking is turned off or on
        self._check_toggle()
        return ret

    def _check_toggle(self) -> None:
        """Helper method to check if a request to toggle the collection state was made

        It checks iof collection state toggle req was made via the server listening on
        rank-0 since last call to report(). Called by report(). Calling this method
        indirectly from report() is the only way to activate the change that is made
        via rank-0
        """
        # If no change just commnunicate the current
        off = self._off
        if self.rank == 0 and self.toggle:
            off = not self._off
            self.toggle = False
        st = torch.tensor(off, dtype=torch.bool, device=self.dev)
        torch.distributed.broadcast(st, 0)  # Blocking
        # save old switch
        off = self._off
        self._off = bool(st.item())
        if off != self._off:
            if not self._off:
                self.start = self.start_method
                self.stop = self.stop_method
                state = "ON"
            else:
                self.start = self.null_method
                self.stop = self.null_method
                state = "OFF"
            if self.rank == 0:
                logger.info(f"Toggling StragglerDetector State {state}")

    def _handler(self) -> None:
        """Thread function for the controller.

        It is a tcp-server that listens on a port. Uses HTTP protocol.
        If connected to it using curl, it indicates a toggle of the
        collection state. The actual toggling happens at the end of
        calling report() when _check_toggle() is called.
        """
        resp = r"HTTP/1.0 200 OK\r\nConnection: Close\r\nContent-length: "

        if self.rank == 0:
            state = "OFF" if self._off else "ON"
            logger.info(
                f"Controller ready to recv " f"commands on port {self.port}. Current state {state}"
            )
            while True and self.sock is not None:
                try:
                    conn, _ = self.sock.accept()
                    _ = conn.recv(1024)
                    self.toggle = True
                    state = "ON" if self._off else "OFF"
                    msg = f"Will turn StragglerDetector {state} at next logging interval"
                    msg_len = len(msg)
                    final_resp = f"{resp}{msg_len}\r\n\r\n{msg}"
                    conn.send(final_resp.encode())
                    conn.close()
                    logger.info(msg)
                except Exception as err:
                    logger.error(f"Error in stragler handler.. {str(err)}")
                    return

    def _controller(self):
        """Installs a controller listener that is used to toggle collection state.

        Called from configure(). Ignored for all ranks other than rank-0
        """
        try:
            if self.rank == 0:
                neth = "0.0.0.0"
                netp = self.port
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((neth, netp))
                self.sock.listen(128)
                self.ctrlr = threading.Thread(
                    target=self._handler, args=(), name="straggler", daemon=True
                )
                self.ctrlr.start()
        except Exception as err:
            logger.warning(f"StragglerDetector cannot be controlled.. {str(err)}")

    def _min_max(
        self,
        ptime: float,
        btime: float,
        temp: float,
        power: float,
        util: float,
        clock: float,
        flops: float,
    ) -> Union[_StragglerData, None]:
        """Helper function to find the min/max values

        Args:
            ptime (float): avg per iteration gpu time
            btime (float): avg per iteration cpu time
            temp (float): gpu temp at the time of reporting
            power (float): gpu power at the time of reporting
            util (float): gpu util at the time of reporting
            clock (float): gpu clock at the time of reporting
            flops (float): estimated flops for the rank

        Returns:
            Union[_StragglerData, None]: It contains the min/max of few metrics and the
                                         corresponding rank it also has sorted list of
                                         all (flops, rank) sorted by flops (aflops)
                                         or returns None if collecton is disabled
        """
        if self._off:
            return None
        # initialize output data object
        o_dt = _StragglerData()

        prof_data: Dict[str, Union[int, float]] = {}
        data_list: List[Dict[str, Union[int, float]]] = []
        prof_data["rank"] = self.rank
        prof_data["time"] = ptime
        prof_data["btime"] = btime
        prof_data["temp"] = temp
        prof_data["power"] = power
        prof_data["util"] = util
        prof_data["clock"] = clock
        prof_data["flops"] = flops

        if self.rank == 0:
            data_list = [prof_data] * self.world

        # this is blocking by default
        torch.distributed.gather_object(prof_data, object_gather_list=data_list, dst=0)

        if self.rank == 0:
            min_ctime = min(data_list, key=lambda k: k["time"])  # elapsed
            max_ctime = max(data_list, key=lambda k: k["time"])  # elapsed

            min_cbatch = min(data_list, key=lambda k: k["btime"])  # batch time
            max_cbatch = max(data_list, key=lambda k: k["btime"])  # batch time

            min_ctemp = min(data_list, key=lambda k: k["temp"])  # temp
            max_ctemp = max(data_list, key=lambda k: k["temp"])  # temp

            min_cpower = min(data_list, key=lambda k: k["power"])  # power
            max_cpower = max(data_list, key=lambda k: k["power"])  # power

            min_cutil = min(data_list, key=lambda k: k["util"])  # gpu util
            max_cutil = max(data_list, key=lambda k: k["util"])  # gpu util

            min_cclock = min(data_list, key=lambda k: k["clock"])  # gpu clock
            max_cclock = max(data_list, key=lambda k: k["clock"])  # gpu clock

            min_val = min_ctime["time"]
            min_rank = min_ctime["rank"]
            max_val = max_ctime["time"]
            max_rank = max_ctime["rank"]
            o_dt.min_elapsed = _ValueWithRank(min_val, int(min_rank), "ms")
            o_dt.max_elapsed = _ValueWithRank(max_val, int(max_rank), "ms")

            min_val = min_cbatch["btime"]
            min_rank = min_cbatch["rank"]
            max_val = max_cbatch["btime"]
            max_rank = max_cbatch["rank"]
            o_dt.min_btime = _ValueWithRank(min_val, int(min_rank), "ms")
            o_dt.max_btime = _ValueWithRank(max_val, int(max_rank), "ms")

            min_val = min_ctemp["temp"]
            min_rank = min_ctemp["rank"]
            max_val = max_ctemp["temp"]
            max_rank = max_ctemp["rank"]
            o_dt.min_temp = _ValueWithRank(min_val, int(min_rank), "C")
            o_dt.max_temp = _ValueWithRank(max_val, int(max_rank), "C")

            min_val = min_cpower["power"]
            min_rank = min_cpower["rank"]
            max_val = max_cpower["power"]
            max_rank = max_cpower["rank"]
            o_dt.min_power = _ValueWithRank(min_val, int(min_rank), "W")
            o_dt.max_power = _ValueWithRank(max_val, int(max_rank), "W")

            min_val = min_cutil["util"]
            min_rank = min_cutil["rank"]
            max_val = max_cutil["util"]
            max_rank = max_cutil["rank"]
            o_dt.min_util = _ValueWithRank(min_val, int(min_rank), "%")
            o_dt.max_util = _ValueWithRank(max_val, int(max_rank), "%")

            min_val = min_cclock["clock"]
            min_rank = min_cclock["rank"]
            max_val = max_cclock["clock"]
            max_rank = max_cclock["rank"]
            o_dt.min_clock = _ValueWithRank(min_val, int(min_rank), "MHz")
            o_dt.max_clock = _ValueWithRank(max_val, int(max_rank), "MHz")

            o_dt.aflops = [
                _ValueWithRank(d.get("flops", 0.0), int(d.get("rank", -1)))
                for _, d in enumerate(data_list)
            ]
            o_dt.aflops.sort(key=lambda val_with_rank: val_with_rank()[0])
        # wait for everyone here
        torch.distributed.barrier()

        return o_dt

    @property
    def enabled(self) -> bool:
        """Can be called to check the enabled state of the instance

        Note:
            After the request to toggle the state, the
            actual state change happens at end of call
            to report()
        """
        return not self._off

    @property
    def configured(self) -> bool:
        """Can be called to check if the the instance is already configured

        Returns:
            bool: returns True if configure was called and was a success, else False
        """
        return StragglerDetector._configured

    @property
    def my_rank(self):
        """Can be called to get configured rank of this instance

        Returns:
            int: Configured rank for this instance
        """
        return self.rank

    @property
    def world_size(self) -> int:
        """Can be called to get configured world of this instance

        Returns:
            int: World size configured for this instance
        """
        return self.world

    def null_method(self) -> None:
        """Default method to initialize start/stop method ptrs"""
        pass

    def __enter__(self) -> "StragglerDetector":
        """Define context/instance entry

        Returns:
            StragglerDetector: the instance
        """
        self.start()
        return self

    def __call__(self, bdata: bool = False) -> "StragglerDetector":
        """Callable for the instance. Set context state,

        Useful when the context is used for cpu timers only when bdata=True

        Args:
            bdata (bool, optional): when true, only enables cpu timers. Defaults to False.

        Returns:
            StragglerDetector: the instance
        """
        self.bdata = bdata
        return self

    def __exit__(
        self,
        ex_type: Optional[Type[BaseException]],
        ex_val: Optional[BaseException],
        ex_tb: Optional[TracebackType],
    ) -> bool:
        """Define context/instance exit, calls the stop method

        Args:
            ex_type (Optional[Type[BaseException]]): Exception type
            ex_val (Optional[BaseException]): _description_
            ex_tb (Optional[TracebackType]): _description_

        Returns:
            bool: True if the exception was handled
        """
        # Should not suppress errors even if turned off
        if ex_type is not None:
            err = traceback.format_exception(ex_type, ex_val, ex_tb)
            logger.warning(f"{str(ex_val)}\n{err}")
        self.stop()
        return False


# Singleton, global visibility
__straggler__ = StragglerDetector()
"""StragglerDetector: private module variable, not be directly accessed
"""