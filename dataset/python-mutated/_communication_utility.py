from chainermn import nccl
import collections
import numpy as np
import pickle
import mpi4py.MPI

def init_ranks(mpi_comm):
    if False:
        while True:
            i = 10
    'Returns rank information of the local process in `mpi_comm`.\n\n    Args:\n        mpi_comm (type:TODO)\n                 MPI Communicator from mpi4py\n\n    Returns:\n        rank_info (list):\n            Elements are:\n                * rank (`mpi_comm.rank`)\n                * intra_rank (rank within the local computing node)\n                * intra_size (number of processes on the node)\n                * inter_rank (rank of the node)\n                * inter_size (number of computing nodes)\n    '
    global_names = mpi_comm.gather(mpi4py.MPI.Get_processor_name())
    if mpi_comm.rank == 0:
        name_to_global_ranks = collections.defaultdict(list)
        for (global_rank, name) in enumerate(global_names):
            name_to_global_ranks[name].append(global_rank)
        for global_ranks in name_to_global_ranks.values():
            global_ranks.sort()
        inter_names = sorted(set(global_names), key=lambda name: name_to_global_ranks[name])
        name_to_inter_rank = {name: inter_rank for (inter_rank, name) in enumerate(inter_names)}
        inter_size = len(inter_names)
        all_ranks = []
        for (global_rank, name) in enumerate(global_names):
            ranks = name_to_global_ranks[name]
            intra_rank = ranks.index(global_rank)
            intra_size = len(ranks)
            inter_rank = name_to_inter_rank[name]
            all_ranks.append((global_rank, intra_rank, intra_size, inter_rank, inter_size))
        my_ranks = mpi_comm.scatter(all_ranks)
    else:
        my_ranks = mpi_comm.scatter(None)
    assert my_ranks[0] == mpi_comm.rank
    return my_ranks

def init_intra_mpi_comm(mpi_comm, intra_rank, inter_rank):
    if False:
        i = 10
        return i + 15
    return mpi_comm.Split(inter_rank, intra_rank)

def init_inter_mpi_comm(mpi_comm, intra_rank, inter_rank):
    if False:
        i = 10
        return i + 15
    return mpi_comm.Split(intra_rank, inter_rank)

def init_nccl_comm(mpi_comm):
    if False:
        i = 10
        return i + 15
    from chainermn import nccl
    if mpi_comm.rank == 0:
        nccl_comm_id = nccl.get_unique_id()
    else:
        nccl_comm_id = None
    nccl_comm_id = mpi_comm.bcast(nccl_comm_id)
    return nccl.NcclCommunicator(mpi_comm.size, nccl_comm_id, mpi_comm.rank)

def inter_allreduce_gpu(inter_mpi_comm, size, gpu_buffer_a, gpu_buffer_b, n_bytes_buffer, n_elems_per_node, n_bytes_per_node, cuda_stream):
    if False:
        i = 10
        return i + 15
    inter_size = inter_mpi_comm.size
    cuda_stream.synchronize()
    inter_mpi_comm.Alltoall([gpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT], [gpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])
    ret = gpu_buffer_a.array(inter_size * n_elems_per_node).reshape(inter_size, n_elems_per_node).sum(axis=0) * (1.0 / size)
    for i in range(0, inter_size):
        gpu_buffer_a.from_device(ret, n_bytes_per_node, i * n_bytes_per_node)
    cuda_stream.synchronize()
    inter_mpi_comm.Alltoall([gpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT], [gpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])
INT_MAX = 2147483647

def chunked_bcast_obj(obj, mpi_comm, max_buf_len=256 * 1024 * 1024, root=0):
    if False:
        print('Hello World!')
    'Split object to max_buf_len size chunks and send them out\n\n    As mpi4py does not accept an object whose pickled size is larger\n    than signed integer max (2147483647) the object is pickled and\n    split into chunks.\n\n    Another hack could be try with mpi_comm.bcast(obj) then rank 0\n    node will receive OverflowError from mpi4py. But in that case rank\n    > 0 nodes shall block busy waiting forever at mpi_comm.bcast(obj).\n\n    Args:\n        obj: A Python object that is to be broadcasted.\n        comm: ChainerMN communicator or MPI4py communicator.\n        root (int): The root process of the scatter operation.\n        max_buf_len (int): Max buffer size to be used at broadcasting\n            binaries. Must not be larger than 2147483647 (INT_MAX).\n            Default value is 256MB.\n    Returns:\n        Broadcasted object.\n\n    '
    assert max_buf_len < INT_MAX
    assert max_buf_len > 0
    assert not (obj is None and mpi_comm.rank == root)
    assert not (obj is not None and mpi_comm.rank != root)
    if obj is not None and mpi_comm.rank == root:
        pickled_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        pickled_bytes = bytearray()
    total_bytes = len(pickled_bytes)
    total_chunk_num = total_bytes // max_buf_len
    if total_bytes % max_buf_len > 0:
        total_chunk_num += 1
    data = mpi_comm.bcast((total_chunk_num, max_buf_len, total_bytes), root=root)
    assert data is not None
    (total_chunk_num, max_buf_len, total_bytes) = data
    for i in range(total_chunk_num):
        b = i * max_buf_len
        e = min(b + max_buf_len, total_bytes)
        if mpi_comm.rank == root:
            buf = pickled_bytes[b:e]
        else:
            buf = bytearray(e - b)
        mpi_comm.Bcast(buf, root=root)
        if mpi_comm.rank != root:
            pickled_bytes[b:e] = buf
    if mpi_comm.rank != root:
        obj = pickle.loads(pickled_bytes)
    return obj

def _get_nccl_type_id(dtype):
    if False:
        while True:
            i = 10
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCCL_FLOAT64
    else:
        raise ValueError('dtype must be float16, float32, or float64.')