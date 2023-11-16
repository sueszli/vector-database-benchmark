import contextlib
import paddle.distributed as dist
from paddle import framework
from paddle.distributed.communication.group import _get_global_group, _warn_cur_rank_not_in_group

class P2POp:
    """
    A class that makes point-to-point operations for "batch_isend_irecv".

    This class creates the type of P2P operation, communication buffer, peer rank,
    Group. Instances of this class will be passed to
    ``paddle.distributed.batch_isend_irecv`` for point-to-point communication.

    Args:
        op (callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``paddle.distributed.isend`` or ``paddle.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): The destination or source rank.
        group (Group, optional): The group instance return by new_group or None for global
            default group. Default: None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> dist.init_parallel_env()
            >>> rank = dist.get_rank()
            >>> world_size = dist.get_world_size()

            >>> send_t = paddle.arange(2) + rank
            >>> # paddle.tensor([0, 1])  # Rank-0
            >>> # paddle.tensor([1, 2])  # Rank-1

            >>> recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)

            >>> send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
            >>> recv_op = dist.P2POp(dist.irecv, recv_t, (rank - 1 + world_size) % world_size)

    """

    def __init__(self, op, tensor, peer, group=None):
        if False:
            print('Hello World!')
        if op not in [dist.isend, dist.irecv]:
            raise RuntimeError('Invalid ``op`` function. Expected ``op`` to be of type ``paddle.distributed.isend`` or ``paddle.distributed.irecv``.')
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = _get_global_group() if group is None else group

@contextlib.contextmanager
def _with_batch_p2p_guard(backend):
    if False:
        while True:
            i = 10
    if backend == 'NCCL':
        framework.core.ProcessGroupNCCL.group_start()
    try:
        yield
    finally:
        if backend == 'NCCL':
            framework.core.ProcessGroupNCCL.group_end()

def _check_p2p_op_list(p2p_op_list):
    if False:
        i = 10
        return i + 15
    '\n    Helper to check that the ``p2p_op_list`` is a list of P2POp instances and\n    all ops use the same backend.\n    '
    if not isinstance(p2p_op_list, list) or not all((isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list)):
        raise RuntimeError('Invalid ``p2p_op_list``. Each op is expected to to be of type ``paddle.distributed.P2POp``.')
    backend = p2p_op_list[0].group.backend
    if not all((backend == p2p_op.group.backend for p2p_op in p2p_op_list)):
        raise RuntimeError('All groups need to use the same backend.')

def batch_isend_irecv(p2p_op_list):
    if False:
        i = 10
        return i + 15
    '\n    Send or Receive a batch of tensors asynchronously and return a list of requests.\n\n    Process each of the point-to-point operations in ``p2p_op_list`` and return the\n    corresponding tasks. NCCL are currently supported.\n\n    Args:\n        p2p_op_list (List[P2POp]): A list of point-to-point operations(type of each operator is\n            ``paddle.distributed.P2POp``). The order of the isend/irecv in the list\n            matters and it needs to match with corresponding isend/irecv on the\n            remote end.\n\n    Returns:\n        A list of distributed tasks returned by calling the corresponding\n        op in the op_list.\n\n    Warning:\n        This API only supports the dygraph mode.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n\n            >>> import paddle\n            >>> import paddle.distributed as dist\n\n            >>> dist.init_parallel_env()\n            >>> rank = dist.get_rank()\n            >>> world_size = dist.get_world_size()\n\n            >>> send_t = paddle.arange(2) + rank\n            >>> # paddle.tensor([0, 1])  # Rank-0\n            >>> # paddle.tensor([1, 2])  # Rank-1\n\n            >>> recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)\n\n            >>> send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)\n            >>> recv_op = dist.P2POp(dist.irecv, recv_t, (rank - 1 + world_size) % world_size)\n\n            >>> tasks = dist.batch_isend_irecv([send_op, recv_op])\n\n            >>> for task in tasks:\n            ...     task.wait()\n\n            >>> print(recv_t)\n            >>> # paddle.tensor([1, 2])     # Rank-0\n            >>> # paddle.tensor([0, 1])     # Rank-1\n    '
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    if _warn_cur_rank_not_in_group(group):
        return
    if framework.in_dynamic_mode():
        group = _get_global_group() if group is None else group
        backend = group.backend
        tasks = []
        with _with_batch_p2p_guard(backend):
            for p2p_op in p2p_op_list:
                op = p2p_op.op
                tensor = p2p_op.tensor
                peer = p2p_op.peer
                comm_group = p2p_op.group
                task = op(tensor, peer, comm_group)
                if task is not None:
                    tasks.append(task)
        return tasks
    else:
        raise RuntimeError("Don't support static graph mode currently.")