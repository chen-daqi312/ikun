"""
infiniccl_bridge — dlopen libinfiniccl.so and expose allreduce for BI-V100 TP

replaces torch.distributed.all_reduce with infinicclAllReduce to bypass
NCCL legacy fence overhead on non-NVLink topologies

usage:
    from ex_engine.python.infiniccl_bridge import init_comm, infiniccl_allreduce
    init_comm(rank, world_size)  # call once per TP worker
    infiniccl_allreduce(tensor)  # drop-in for tensor_model_parallel_all_reduce
"""

import ctypes
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_lib: Optional[ctypes.CDLL] = None
_comm = None
_rank = -1
_world_size = -1


def _find_and_load() -> Optional[ctypes.CDLL]:
    search = [
        os.path.join(os.path.dirname(__file__), "..", "prebuilt", "libinfiniccl.so"),
        "/usr/local/corex/lib64/libinfiniccl.so",
    ]
    try:
        import vllm
        vllm_root = os.path.dirname(vllm.__file__)
        search.insert(0, os.path.join(vllm_root, "libinfiniccl.so"))
        search.insert(1, os.path.join(vllm_root, "..", "ex_engine", "prebuilt", "libinfiniccl.so"))
    except ImportError:
        pass

    # preload libtorch_cuda so infiniccl can resolve nccl symbols
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        ctypes.CDLL(os.path.join(torch_lib, "libtorch_cuda.so"), mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        logger.warning("failed to preload libtorch_cuda: %s", e)

    for path in search:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            if hasattr(lib, "infinicclAllReduce"):
                logger.info("infiniccl loaded from %s", path)
                return lib
        except Exception as e:
            logger.warning("failed to load infiniccl from %s: %s", path, e)
    return None


def init_comm(rank: int, world_size: int) -> bool:
    global _lib, _comm, _rank, _world_size
    if _lib is None:
        _lib = _find_and_load()
    if _lib is None:
        logger.warning("infiniccl not available, falling back to torch.distributed")
        return False

    _rank = rank
    _world_size = world_size

    import torch.distributed as dist

    # rank 0 generates UniqueId, broadcast to all ranks via torch.distributed
    uid_buf = ctypes.create_string_buffer(128)
    if rank == 0:
        ret = _lib.infinicclGetUniqueId(uid_buf)
        if ret != 0:
            logger.warning("infinicclGetUniqueId failed: %d", ret)
            return False

    # broadcast the 128-byte uid from rank 0 to all ranks
    uid_tensor = torch.frombuffer(uid_buf, dtype=torch.uint8).clone().cuda()
    dist.broadcast(uid_tensor, src=0)
    # copy back to uid_buf
    uid_bytes = uid_tensor.cpu().numpy().tobytes()
    ctypes.memmove(uid_buf, uid_bytes, 128)

    comm_ptr = ctypes.c_void_p()
    ret = _lib.infinicclCommInitRank(
        ctypes.byref(comm_ptr), world_size, uid_buf, rank
    )
    if ret != 0:
        logger.warning("infinicclCommInitRank failed: %d rank=%d", ret, rank)
        return False

    _comm = comm_ptr
    print(f"[infiniccl] comm initialized rank={rank}/{world_size}",
          file=__import__('sys').stderr, flush=True)
    return True


def infiniccl_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    global _comm, _rank, _world_size
    # lazy init: first call triggers comm setup from torch.distributed
    if _lib is not None and _comm is None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                _rank = dist.get_rank()
                _world_size = dist.get_world_size()
                init_comm(_rank, _world_size)
        except Exception as e:
            logger.warning("infiniccl lazy init failed: %s", e)

    if _lib is None or _comm is None:
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        return tensor_model_parallel_all_reduce(tensor)

    if tensor.dtype == torch.float16:
        dtype_enum = 0  # infinicclFloat16
    elif tensor.dtype == torch.float32:
        dtype_enum = 1  # infinicclFloat32
    else:
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        return tensor_model_parallel_all_reduce(tensor)

    out = torch.empty_like(tensor)
    stream = torch.cuda.current_stream().cuda_stream

    ret = _lib.infinicclAllReduce(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        tensor.numel(),
        dtype_enum,  # data type
        0,  # infinicclSum
        _comm,
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        logger.warning("infinicclAllReduce failed (%d), falling back", ret)
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        return tensor_model_parallel_all_reduce(tensor)

    return out
