"""
infiniccl_bridge — dlopen libinfiniccl.so and expose allreduce for BI-V100 TP
"""

import ctypes
import os
import sys

import torch
import torch.distributed as dist

_lib = None
_comm = None
_rank = -1
_world_size = -1


def _find_and_load():
    import vllm
    vllm_root = os.path.dirname(vllm.__file__)
    search = [
        os.path.join(vllm_root, "..", "ex_engine", "prebuilt", "libinfiniccl.so"),
        os.path.join(vllm_root, "libinfiniccl.so"),
        os.path.join(os.path.dirname(__file__), "..", "prebuilt", "libinfiniccl.so"),
        "/usr/local/corex/lib64/libinfiniccl.so",
    ]

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    ctypes.CDLL(os.path.join(torch_lib, "libtorch_cuda.so"), mode=ctypes.RTLD_GLOBAL)

    for path in search:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            lib = ctypes.CDLL(path)
            assert hasattr(lib, "infinicclAllReduce"), f"{path} missing infinicclAllReduce"
            print(f"[infiniccl] loaded from {path}", file=sys.stderr, flush=True)
            return lib

    raise FileNotFoundError(f"libinfiniccl.so not found in {search}")


def init_comm(rank, world_size):
    global _lib, _comm, _rank, _world_size
    if _lib is None:
        _lib = _find_and_load()

    _rank = rank
    _world_size = world_size

    # set cuda device for this rank before nccl comm init
    torch.cuda.set_device(rank)

    uid_buf = ctypes.create_string_buffer(128)
    if rank == 0:
        ret = _lib.infinicclGetUniqueId(uid_buf)
        assert ret == 0, f"infinicclGetUniqueId failed: {ret}"

    # broadcast uid via TP group (all TP ranks must participate)
    from vllm.distributed.parallel_state import get_tp_group
    tp = get_tp_group()
    uid_tensor = torch.tensor(list(uid_buf.raw), dtype=torch.uint8).cuda()
    dist.broadcast(uid_tensor, src=tp.ranks[0], group=tp.device_group)
    uid_bytes = bytes(uid_tensor.cpu().tolist())
    ctypes.memmove(uid_buf, uid_bytes, 128)

    print(f"[infiniccl] CommInitRank rank={rank}/{world_size} device={torch.cuda.current_device()}",
          file=sys.stderr, flush=True)

    comm_ptr = ctypes.c_void_p()
    ret = _lib.infinicclCommInitRank(
        ctypes.byref(comm_ptr), world_size, uid_buf, rank
    )
    assert ret == 0, f"infinicclCommInitRank failed: {ret} rank={rank}"

    _comm = comm_ptr
    print(f"[infiniccl] comm initialized rank={rank}/{world_size}",
          file=sys.stderr, flush=True)


def infiniccl_allreduce(tensor):
    assert _comm is not None, "infiniccl comm not initialized — call init_comm from load_weights first"

    if tensor.dtype == torch.float16:
        dtype_enum = 0
    elif tensor.dtype == torch.float32:
        dtype_enum = 1
    else:
        raise TypeError(f"infiniccl_allreduce unsupported dtype: {tensor.dtype}")

    out = torch.empty_like(tensor)
    stream = torch.cuda.current_stream().cuda_stream

    ret = _lib.infinicclAllReduce(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        tensor.numel(),
        dtype_enum,
        0,  # infinicclSum
        _comm,
        ctypes.c_void_p(stream),
    )
    assert ret == 0, f"infinicclAllReduce failed: {ret}"
    return out
