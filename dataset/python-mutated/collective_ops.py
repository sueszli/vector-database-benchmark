"""TensorFlow collective Ops."""
from tensorflow.python.ops import gen_collective_ops

def all_reduce(t, group_size, group_key, instance_key, merge_op='Add', final_op='Id', subdiv_offsets=(0,), communication_hint='auto', timeout=0):
    if False:
        print('Hello World!')
    "Reduces tensors collectively, across devices.\n\n  Args:\n    t: the tensor to be reduced.\n    group_size: the total number of tensors to be collectively reduced.\n      Each must reside on a different device.  Should be a positive integer.\n    group_key: an integer identifying the group of devices.\n    instance_key: an integer identifying the participating group of Ops.\n    merge_op: string naming the binary Op to be applied to compute each\n      partial reduction.\n    final_op: string naming the unary Op to be applied to each fully\n      reduced value.  Can be 'Id' for no operation.\n    subdiv_offsets: a list of integer offsets into the tensor at which each\n      independent subdivision should begin.  Use [0] if no subdivision should\n      be done.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: a float. If set to a non zero, set a completion timeout to detect\n      staleness.  If the timer goes off, a DeadlineExceededError is raised.  The\n      timeout value in seconds. This feature is experimental.\n\n  Returns:\n    An Op implementing the distributed reduction.\n\n  Raises:\n    ValueError: if any of the input parameter constraints are not met.\n  "
    if group_size < 1:
        raise ValueError(f'Parameter `group_size` to all_reduce must be at least 1. Received: {group_size}.')
    return gen_collective_ops.collective_reduce(t, group_size=group_size, group_key=group_key, instance_key=instance_key, merge_op=merge_op, final_op=final_op, subdiv_offsets=subdiv_offsets, communication_hint=communication_hint.lower(), timeout_seconds=timeout)

def assign_group_v2(group_assignment, device_index, base_key):
    if False:
        print('Hello World!')
    'Assign group key based on group_assignment.\n\n  Args:\n    group_assignment: a 2 dimensional integer Tensor that encodes which devices\n      belong to the same group. The values are indices of the devices within 0\n      to number of devices.\n    device_index: integer for the index of the current device\n    base_key: integer to offset the resulted group_key. The base key shall be\n      unique for different values of group_assignment in the same tf.function.\n  Notes: The device_index argument must be consistent with the index of the\n    device of this Op in the device assignment list. The behavior of this Op is\n    undefined if they are inconsistent.\n\n  Returns:\n    group_size, group_key: The group size and group key for the current device.\n  '
    (group_size, group_key) = gen_collective_ops.collective_assign_group_v2(group_assignment=group_assignment, device_index=device_index, base_key=base_key)
    return (group_size, group_key)

def all_reduce_v2(t, group_size, group_key, instance_key, merge_op='Add', final_op='Id', communication_hint='auto', timeout=0, ordering_token=None, max_subdivs_per_device=-1, name=None):
    if False:
        return 10
    "Reduces tensors collectively, across devices.\n\n  Args:\n    t: the tensor to be reduced.\n    group_size: an int32 tensor. The total number of tensors to be collectively\n      reduced.  Each must reside on a different device.  Should be a positive\n      integer.\n    group_key: an int32 tensor identifying the group of devices.\n    instance_key: an int32 tensor identifying the participating group of Ops.\n    merge_op: string naming the binary Op to be applied to compute each partial\n      reduction.\n    final_op: string naming the unary Op to be applied to each fully reduced\n      value.  Can be 'Id' for no operation.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: a float. If set to a non zero, set a completion timeout to detect\n      staleness.  If the timer goes off, a DeadlineExceededError is raised.  The\n      timeout value in seconds. This feature is experimental.\n    ordering_token: a resource tensor on the same device as the op to order\n      the collectives in a per-device manner by auto control dependency.\n      This argument can be omited when there is one collective Op per\n      `tf.function`, or when explicit control dependency is used instead of\n      auto control dependency.\n    max_subdivs_per_device: int specifying the maximum number of subdivisions a\n      tensor on a device can be divided into. The runtime uses this contraint to\n      parallelize processing of each per-device tensor. Setting to -1 disables\n      subdivision and reverts to previous behavior of not sub-dividing tensor.\n      Setting to 0 uses sytem defaults.\n    name: name of the Op.\n\n  Returns:\n    An Op implementing the distributed reduction.\n  "
    if ordering_token is not None:
        ordering_token = [ordering_token]
    else:
        ordering_token = []
    return gen_collective_ops.collective_reduce_v2(t, group_size=group_size, group_key=group_key, instance_key=instance_key, merge_op=merge_op, final_op=final_op, communication_hint=communication_hint.lower(), timeout_seconds=timeout, is_stateless=False, ordering_token=ordering_token, max_subdivs_per_device=max_subdivs_per_device, name=name)

def all_gather(t, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    if False:
        while True:
            i = 10
    'Accumulates tensors collectively, across devices, along first dimension.\n\n  Args:\n    t: the tensor to participate in the accumulation.\n    group_size: the total number of tensors to be collectively accumulated.\n      Each must reside on a different device. Should be a positive integer.\n    group_key: an integer identifying the group of devices.\n    instance_key: an integer identifying the participating group of Ops.\n    communication_hint: preferred collective communication. The implementation\n      may fall back to another mechanism. Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: a float. If set to a non zero, set a completion timeout to detect\n      staleness. If the timer goes off, a DeadlineExceededError is raised. The\n      timeout value in seconds. This feature is experimental.\n\n  Returns:\n    An Op implementing the distributed operation.\n\n  Raises:\n    ValueError: if any of the input parameter constraints are not met.\n  '
    if group_size < 1:
        raise ValueError(f'Parameter `group_size` to all_gather must be at least 1. Received: {group_size}.')
    return gen_collective_ops.collective_gather(t, shape=[0], group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)

def all_gather_v2(t, group_size, group_key, instance_key, communication_hint='auto', timeout=0, ordering_token=None, name=None):
    if False:
        i = 10
        return i + 15
    'Accumulates tensors collectively, across devices, along first dimension.\n\n  Args:\n    t: the tensor to participate in the accumulation.\n    group_size: an int32 tensor, the total number of tensors to be collectively\n      accumulated. Each must reside on a different device. Should be a positive\n      integer.\n    group_key: an int32 tensor identifying the group of devices.\n    instance_key: an int32 tensor identifying the participating group of Ops.\n    communication_hint: preferred collective communication. The implementation\n      may fall back to another mechanism. Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: a float. If set to a non zero, set a completion timeout to detect\n      staleness. If the timer goes off, a DeadlineExceededError is raised. The\n      timeout value in seconds. This feature is experimental.\n    ordering_token: a resource tensor on the same device as the op to order\n      the collectives in a per-device manner by auto control dependency.\n      This argument can be omited when there is one collective Op per\n      `tf.function`, or when explicit control dependency is used instead of\n      auto control dependency.\n    name: name of the Op.\n\n  Returns:\n    An Op implementing the distributed operation.\n  '
    if ordering_token is not None:
        ordering_token = [ordering_token]
    else:
        ordering_token = []
    return gen_collective_ops.collective_gather_v2(t, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout, is_stateless=False, ordering_token=ordering_token, name=name)

def broadcast_send(t, shape, dtype, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    if False:
        while True:
            i = 10
    'Broadcasts one tensor to a group of others, across devices.\n\n  Args:\n    t: the tensor to be sent.\n    shape: the shape of the tensor being sent, which must agree with t.\n    dtype: the type of the tensor being sent, which must agree with t.\n    group_size: one plus the number of receiving tensors, i.e. the total\n      number of devices participating.  Each tensor must reside on a\n      different device.\n    group_key: an integer identifying the group of devices.\n    instance_key: an integer identifying the participating group of Ops.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: If set to a non zero, set a completion timeout to detect staleness.\n      If the timer goes off, a DeadlineExceededError is raised.\n      The timeout value in seconds. This feature is experimental.\n\n  Returns:\n    An Op implementing the distributed broadcast send.\n\n  Raises:\n    ValueError: if any of the input parameter constraints are not met.\n\n  Note that the shape and dtype arguments appear redundant since they\n  should be obtainable from t.  The are two reasons for including\n  them.  First, the shape and type of tensors passed via broadcast must\n  be known ahead of time in their most specific form so that the receive\n  side can allocate memory for the operation and shape/type inference can\n  carry forward from there.  Including the same declarations on the\n  send side clarifies a commitment already made.  Secondly, having nearly\n  identical use syntax for send and receive sides may simplify tool-driven\n  generation of broadcast.\n  '
    if group_size <= 1:
        raise ValueError(f'Parameter `group_size` to broadcast_send must be at least 2. Received: {group_size}.')
    if t.shape != shape:
        raise ValueError(f'Shape of broadcast_send tensor `t` not equal to declared shape. Received {t.shape}, expected {shape}.')
    if t.dtype != dtype:
        raise ValueError(f'Type of broadcast_send tensor `t` not equal to declared type. Received {t.dtype}, expected {dtype}.')
    return gen_collective_ops.collective_bcast_send(t, shape=shape, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)

def broadcast_send_v2(t, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    if False:
        while True:
            i = 10
    'Broadcasts one tensor to a group of others, across devices.\n\n  Args:\n    t: the tensor to be sent.\n    group_size: an int32 tensor.  One plus the number of receiving tensors, i.e.\n        the total number of devices participating.  Each tensor must reside on a\n        different device.\n    group_key: an int32 tensor identifying the group of devices.\n    instance_key: an int32 tensor identifying the participating group of Ops.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: If set to a non zero, set a completion timeout to detect staleness.\n      If the timer goes off, a DeadlineExceededError is raised.\n      The timeout value in seconds. This feature is experimental.\n\n  Returns:\n    An Op implementing the distributed broadcast send.\n  '
    return gen_collective_ops.collective_bcast_send_v2(t, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)

def broadcast_recv(shape, dtype, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    if False:
        for i in range(10):
            print('nop')
    'Receives a broadcasts tensor, across devices.\n\n  Args:\n    shape: Shape of the tensor to be received.\n    dtype: Type of the tensor to be received.\n    group_size: one plus the number of receiving tensors, i.e. the total\n      number of devices participating.  Each tensor must reside on a\n      different device.\n    group_key: an integer identifying the group of devices.\n    instance_key: an integer identifying the participating group of Ops.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: If set to a non zero, set a completion timeout to detect staleness.\n      If the timer goes off, a DeadlineExceededError is raised.\n      The timeout value in seconds. This feature is experimental.\n\n  Returns:\n    An Op implementing the broadcast receive.\n\n  Raises:\n    ValueError: if any of the input parameter constraints are not met.\n  '
    if group_size <= 1:
        raise ValueError(f'Parameter `group_size` to broadcast_send must be at least 2. Received: {group_size}.')
    return gen_collective_ops.collective_bcast_recv(shape=shape, T=dtype, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)

def broadcast_recv_v2(shape, dtype, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    if False:
        i = 10
        return i + 15
    'Receives a broadcasts tensor, across devices.\n\n  Args:\n    shape: an int tensor.  Shape of the tensor to be received.\n    dtype: Type of the tensor to be received.\n    group_size: an int32 tensor.  One plus the number of receiving tensors, i.e.\n        the total number of devices participating.  Each tensor must reside on a\n        different device.\n    group_key: an int32 tensor identifying the group of devices.\n    instance_key: an int32 tensor identifying the participating group of Ops.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout: If set to a non zero, set a completion timeout to detect staleness.\n      If the timer goes off, a DeadlineExceededError is raised.\n      The timeout value in seconds. This feature is experimental.\n\n  Returns:\n    An Op implementing the broadcast receive.\n  '
    return gen_collective_ops.collective_bcast_recv_v2(T=dtype, group_size=group_size, group_key=group_key, instance_key=instance_key, shape=shape, communication_hint=communication_hint.lower(), timeout_seconds=timeout)

def initialize_communicator(group_key, rank, group_size, communication_hint='auto', timeout_seconds=0):
    if False:
        print('Hello World!')
    'Initializes a collective communicator.\n\n  This creates a collective communicator, which represents membership to a\n  collective group identified by the group_key. It should be called once per\n  member of the group, and each member needs to be on a different device.\n  It blocks until all members of the group run this op.\n\n  Communicators of a group can only be initialized once. Trying to initialize\n  communicators for an existing group key will result in an error.\n\n  Args:\n    group_key: an int32 `tf.Tensor` identifying the group.\n    rank: an `tf.Tensor` specifying the rank of this device in the group. If\n      specified, the rank is required to be unique in the group.\n    group_size: an int32 `tf.Tensor`. The size of the group.\n    communication_hint: preferred collective communication.  The implementation\n      may fall back to another mechanism.  Options include `auto`, `ring`, and\n      `nccl`.\n    timeout_seconds: If set to a non zero, set a completion timeout to detect\n      staleness. If the timer goes off, a DeadlineExceededError is raised. The\n      timeout value in seconds. This feature is experimental.\n\n\n  Returns:\n    A resource `tf.Tensor`.\n  '
    return gen_collective_ops.collective_initialize_communicator(group_key=group_key, rank=rank, group_size=group_size, communication_hint=communication_hint, timeout_seconds=timeout_seconds)

def all_reduce_v3(communicator, t, reduction='Add', group_assignment=None, timeout_seconds=None):
    if False:
        i = 10
        return i + 15
    'Reduces tensors mutually.\n\n  Args:\n    communicator: the resource `tf.Tensor` returned from\n      `initialize_communicator`.\n    t: the `tf.Tensor` to be reduced.\n    reduction: a string. The name of the operation to reduce the values.\n      Accpeted values are `"min"`, `"max"`, `"mul"`, `"add"`.\n    group_assignment: Optional int32 `tf.Tensor` with shape [num_groups,\n      num_ranks_per_group]. `group_assignment[i]` represents the ranks in the\n      `ith` subgroup.\n    timeout_seconds: If set to a non zero, set a completion timeout to detect\n      staleness. If the timer goes off, a DeadlineExceededError is raised. The\n      timeout value in seconds. This feature is experimental.\n\n  Returns:\n    The reduced `tf.Tensor`.\n  '
    if group_assignment is None:
        group_assignment = []
    return gen_collective_ops.collective_reduce_v3(communicator=communicator, input=t, group_assignment=group_assignment, reduction=reduction, timeout_seconds=timeout_seconds)

def all_to_all_v2(t, group_size, group_key, instance_key, communication_hint='auto', timeout=0, ordering_token=None, name=None):
    if False:
        return 10
    'Exchanges tensors mutually.\n\n  Args:\n    t: a `tf.Tensor`. The first dimension should have the length as the size of\n      the group. `t[i]` is sent to `rank i` within the group.\n    group_size: an int32 tensor, the total number of tensors to be mutually\n      exchanged. Each must reside on a different device. Should be a positive\n      integer.\n    group_key: an int32 tensor identifying the group of devices.\n    instance_key: an int32 tensor identifying the participating group of Ops.\n    communication_hint: preferred collective communication. The implementation\n      may fall back to another mechanism. Options include `auto` and `nccl`.\n    timeout: a float. If set to a non zero, set a completion timeout to detect\n      staleness. If the timer goes off, a DeadlineExceededError is raised. The\n      timeout value in seconds. This feature is experimental.\n    ordering_token: a resource tensor on the same device as the op to order the\n      collectives in a per-device manner by auto control dependency. This\n      argument can be omited when there is one collective Op per `tf.function`,\n      or when explicit control dependency is used instead of auto control\n      dependency.\n    name: name of the Op.\n\n  Returns:\n    An Op implementing the distributed operation.\n  '
    if ordering_token is not None:
        ordering_token = [ordering_token]
    else:
        ordering_token = []
    return gen_collective_ops.collective_all_to_all_v2(t, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout, is_stateless=False, ordering_token=ordering_token, name=name)

def all_to_all_v3(communicator, t, group_assignment=None, timeout_seconds=None):
    if False:
        print('Hello World!')
    'Exchanges tensors mutually.\n\n  Args:\n    communicator: the resource `tf.Tensor` returned from\n      `initialize_communicator`.\n    t: a `tf.Tensor`. The first dimension should have the length as the size of\n      the group. `t[i]` is sent to `rank i` within the group.\n    group_assignment: Optional int32 `tf.Tensor` with shape [num_groups,\n      num_ranks_per_group]. `group_assignment[i]` represents the ranks in the\n      `ith` subgroup.\n    timeout_seconds: If set to a non zero, set a completion timeout to detect\n      staleness. If the timer goes off, a DeadlineExceededError is raised. The\n      timeout value in seconds. This feature is experimental.\n\n  Returns:\n    a `tf.Tensor`. `t[i]` is sent from `rank i` within the group.\n  '
    if group_assignment is None:
        group_assignment = []
    return gen_collective_ops.collective_all_to_all_v3(communicator=communicator, input=t, group_assignment=group_assignment, timeout_seconds=timeout_seconds)