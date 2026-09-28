"""
infiniccl_bridge — nccl allreduce via ctypes for BI-V100 TP

creates a second nccl communicator (independent from torch.distributed)
using proper struct-by-value passing for ncclUniqueId. uses this comm
for allreduce in the fused AR path.
"""

import ctypes
import os
import sys

import torch
import torch.distributed as dist


class NcclUniqueId(ctypes.Structure):
    _fields_ = [('internal', ctypes.c_char * 128)]


_nccl_lib = None
_comm = None
_rank = -1
_world_size = -1


def _load_nccl():
    global _nccl_lib
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    _nccl_lib = ctypes.CDLL(os.path.join(torch_lib, "libtorch_cuda.so"), mode=ctypes.RTLD_GLOBAL)

    _nccl_lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
    _nccl_lib.ncclGetUniqueId.restype = ctypes.c_int
    _nccl_lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, NcclUniqueId, ctypes.c_int
    ]
    _nccl_lib.ncclCommInitRank.restype = ctypes.c_int
    _nccl_lib.ncclAllReduce.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    ]
    _nccl_lib.ncclAllReduce.restype = ctypes.c_int

    print("[infiniccl] nccl symbols loaded from libtorch_cuda.so", file=sys.stderr, flush=True)


def _find_and_load():
    """compat — just load nccl"""
    _load_nccl()
    return _nccl_lib


def init_comm(rank, world_size):
    global _comm, _rank, _world_size
    if _nccl_lib is None:
        _load_nccl()

    _rank = rank
    _world_size = world_size
    torch.cuda.set_device(rank)

    uid = NcclUniqueId()
    if rank == 0:
        ret = _nccl_lib.ncclGetUniqueId(ctypes.byref(uid))
        assert ret == 0, f"ncclGetUniqueId failed: {ret}"

    # broadcast uid via TP group
    from vllm.distributed.parallel_state import get_tp_group
    tp = get_tp_group()
    uid_tensor = torch.tensor(list(uid.internal), dtype=torch.uint8).cuda()
    dist.broadcast(uid_tensor, src=tp.ranks[0], group=tp.device_group)
    uid.internal = bytes(uid_tensor.cpu().tolist())

    print(f"[infiniccl] CommInitRank rank={rank}/{world_size} device={torch.cuda.current_device()}",
          file=sys.stderr, flush=True)

    comm_ptr = ctypes.c_void_p()
    ret = _nccl_lib.ncclCommInitRank(ctypes.byref(comm_ptr), world_size, uid, rank)
    assert ret == 0, f"ncclCommInitRank failed: {ret} rank={rank}"

    _comm = comm_ptr
    print(f"[infiniccl] comm initialized rank={rank}/{world_size}",
          file=sys.stderr, flush=True)


def infiniccl_allreduce(tensor):
    assert _comm is not None, "comm not initialized"

    # ncclDataType: 6=ncclFloat16, 7=ncclFloat32
    if tensor.dtype == torch.float16:
        nccl_dtype = 6
    elif tensor.dtype == torch.float32:
        nccl_dtype = 7
    else:
        raise TypeError(f"unsupported dtype: {tensor.dtype}")

    out = torch.empty_like(tensor)
    stream = torch.cuda.current_stream().cuda_stream

    ret = _nccl_lib.ncclAllReduce(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        tensor.numel(),
        nccl_dtype,
        0,  # ncclSum
        _comm,
        ctypes.c_void_p(stream),
    )
    assert ret == 0, f"ncclAllReduce failed: {ret}"
    return out
