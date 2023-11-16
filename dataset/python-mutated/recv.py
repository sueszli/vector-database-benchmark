from paddle.distributed.communication import stream

def recv(tensor, src=0, group=None, sync_op=True):
    if False:
        print('Hello World!')
    '\n    Receive a tensor to the sender.\n\n    Args:\n        tensor (Tensor): The tensor to receive. Its data type\n            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.\n        src (int): The source rank id.\n        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.\n        sync_op (bool, optional): Whether this op is a sync op. The default value is True.\n\n    Returns:\n        Return a task object.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n            >>> import paddle\n            >>> import paddle.distributed as dist\n\n            >>> dist.init_parallel_env()\n            >>> if dist.get_rank() == 0:\n            ...     data = paddle.to_tensor([7, 8, 9])\n            ...     dist.send(data, dst=1)\n            >>> else:\n            ...     data = paddle.to_tensor([1, 2, 3])\n            ...     dist.recv(data, src=0)\n            >>> print(data)\n            >>> # [7, 8, 9] (2 GPUs)\n    '
    return stream.recv(tensor, src=src, group=group, sync_op=sync_op, use_calc_stream=False)

def irecv(tensor, src=None, group=None):
    if False:
        while True:
            i = 10
    '\n    Receive a tensor to the sender.\n\n    Args:\n        tensor (Tensor): The Tensor to receive. Its data type\n            should be float16, float32, float64, int32, int64, int8, uint8, bool or bfloat16.\n        src (int): The source rank id.\n        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.\n\n    Returns:\n        Return a task object.\n\n    Warning:\n        This API only supports the dygraph mode.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env: DISTRIBUTED)\n            >>> import paddle\n            >>> import paddle.distributed as dist\n\n            >>> dist.init_parallel_env()\n            >>> if dist.get_rank() == 0:\n            ...     data = paddle.to_tensor([7, 8, 9])\n            ...     task = dist.isend(data, dst=1)\n            >>> else:\n            ...     data = paddle.to_tensor([1, 2, 3])\n            ...     task = dist.irecv(data, src=0)\n            >>> task.wait()\n            >>> print(data)\n            >>> # [7, 8, 9] (2 GPUs)\n    '
    return recv(tensor, src, group, sync_op=False)