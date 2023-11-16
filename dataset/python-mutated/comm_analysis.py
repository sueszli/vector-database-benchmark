import math
from enum import IntEnum
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V

class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2

class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2

def get_gpu_type() -> NVIDIA_GPU_TYPE:
    if False:
        return 10
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run)
    if 'V100' in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif 'A100' in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif 'H100' in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    else:
        return NVIDIA_GPU_TYPE.AMPERE

def get_collective_type(snode: 'BaseSchedulerNode') -> NCCL_COLL:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(snode.node, (ir.AllReduce, ir.AllReduceCoalesced)):
        return NCCL_COLL.ALL_REDUCE
    elif isinstance(snode.node, (ir.AllGatherIntoTensor, ir.AllGatherIntoTensorCoalesced)):
        return NCCL_COLL.ALL_GATHER
    elif isinstance(snode.node, (ir.ReduceScatterTensor, ir.ReduceScatterTensorCoalesced)):
        return NCCL_COLL.REDUCE_SCATTER
    else:
        raise Exception(f'Unsupported collective type: {snode.node}')

class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2

class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1

class NCCL_PROTO(IntEnum):
    LL = 0
baseLat = torch.tensor([[6.8], [6.6]])
hwLat = torch.tensor([[[0.6], [0.6]], [[1.0], [1.0]], [[5.0], [2.7]]])
llMaxBws = torch.tensor([[39.0, 39.0, 20.4], [87.7, 22.5, 19.0], [87.7, 22.5, 19.0]])

def estimate_nccl_collective_runtime(snode: 'BaseSchedulerNode') -> float:
    if False:
        while True:
            i = 10
    '\n    Returns estimated NCCL collective runtime in nanoseconds (ns).\n\n    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.\n    We aim to estimate the runtime as accurately as possible.\n\n    Assumptions:\n    - only ring algorithm (NCCL_ALGO_RING) is used\n    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used\n    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.\n    - collective is one of: allreduce, reducescatter, allgather\n    '
    tensor_numel = V.graph.sizevars.size_hint(sympy_product(snode.node.layout.size))
    tensor_dtype = snode.node.layout.dtype
    tensor_storage_size_bytes = tensor_numel * get_dtype_size(tensor_dtype)
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024
    num_gpus_per_node = 8
    (_, _, group_size) = snode.node.constant_args
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size
    if nRanks <= 1:
        return 0
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(snode)
    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw
    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2].item()
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2
    busBw = nChannels * bw
    busBw = min(llMaxBw, busBw * (1.0 / 4.0 if nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE else 1.0 / 3.0))
    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1
    ratio = 1.0 * nRanks / nsteps
    bandwidth = busBw * ratio
    bandwidth_GB_per_ns = bandwidth / 1000000000.0
    intraHw = NCCL_HW.NVLINK
    hw = intraHw if nNodes == 1 else NCCL_HW.NET
    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nInterSteps = nNodes - 1
    latency = baseLat[nccl_algo][nccl_proto].item()
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto].item()
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto].item()
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat
    latency_ns = latency * 1000.0
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    return transport_ns + latency_ns