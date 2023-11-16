import datetime
import logging
import os
import shutil
import time
import numpy
import pygloo
import ray
from ray._private import ray_constants
from ray.util.collective.collective_group import gloo_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import AllGatherOptions, AllReduceOptions, Backend, BarrierOptions, BroadcastOptions, RecvOptions, ReduceOptions, ReduceScatterOptions, SendOptions
logger = logging.getLogger(__name__)

class Rendezvous:
    """A rendezvous class for different actor/task processes to meet.

    To initialize an GLOO collective communication group, different
    actors/tasks spawned in Ray in a collective group needs to meet
    each other to synchronize the GLOOUniqueID. This class guarantees
    they meet via the GLOOUniqueIDStore, initialized on the rank=0
    process.

    Args:
        group_name: the unique user-specified group name.
    """

    def __init__(self, group_name, context, store_type, device_type):
        if False:
            while True:
                i = 10
        self._group_name = group_name
        self._context = context
        redis_address = ray._private.worker._global_node.redis_address
        (self._redis_ip_address, self._redis_port) = redis_address.split(':') if store_type == 'redis' else (None, None)
        self._process_ip_address = ray.util.get_node_ip_address()
        logger.debug('Redis address: {}, port: {}, this actor address: {}.'.format(self._redis_ip_address, self._redis_port, self._process_ip_address))
        self._store_type = store_type
        self._device_type = device_type
        self._store = None
        self._device = None
        self.create_store(store_type)
        self.create_device(device_type)

    def create_store(self, store_type):
        if False:
            while True:
                i = 10
        if store_type == 'ray_internal_kv':
            ray_internal_kv_store = gloo_util.RayInternalKvStore(self._group_name)
            self._store = pygloo.rendezvous.CustomStore(ray_internal_kv_store)
        elif store_type == 'redis':
            redisStore = pygloo.rendezvous.RedisStore(self._redis_ip_address, int(self._redis_port))
            redis_password = ray._private.worker._global_node.redis_password
            if redis_password is None or len(redis_password) == 0:
                redis_password = ray_constants.REDIS_DEFAULT_PASSWORD
            redisStore.authorize(redis_password)
            self._store = redisStore
        elif store_type == 'file':
            store_name = get_store_name(self._group_name)
            store_path = gloo_util.get_gloo_store_path(store_name)
            if self._context.rank == 0:
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                elif os.listdir(store_path) and os.listdir(store_path):
                    shutil.rmtree(store_path)
                    os.makedirs(store_path)
            else:
                while not os.path.exists(store_path):
                    time.sleep(0.1)
            fileStore = pygloo.rendezvous.FileStore(store_path)
            self._store = pygloo.rendezvous.PrefixStore(self._group_name, fileStore)
        elif store_type == 'hash':
            raise NotImplementedError('No implementation for hash store.')
        else:
            raise RuntimeError('Unrecognized store type: {}.'.format(store_type))

    def create_device(self, device_type):
        if False:
            i = 10
            return i + 15
        if device_type == 'tcp':
            attr = pygloo.transport.tcp.attr(self._process_ip_address)
            self._device = pygloo.transport.tcp.CreateDevice(attr)
        elif device_type == 'uv':
            raise NotImplementedError('No implementation for uv.')

    def meet(self, timeout_s=180):
        if False:
            for i in range(10):
                print('nop')
        'Meet at the named actor store.\n\n        Args:\n            timeout_s: timeout in seconds.\n\n        Return:\n            None\n        '
        if timeout_s <= 0:
            raise ValueError("The 'timeout' argument must be positive. Got '{}'.".format(timeout_s))
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        (q, s) = (None, None)
        if self._store_type == 'redis' or self._store_type == 'ray_internal_kv':
            while elapsed < timeout_delta:
                try:
                    q = ray.get_actor('gloo_queue')
                    s = ray.get_actor(f'gloo_{self._group_name}_signal')
                    break
                except ValueError:
                    if self._context.rank == 0:
                        if not q:
                            ray.remote(gloo_util.glooQueue).options(name='gloo_queue', lifetime='detached').remote(1000)
                        if not s:
                            gloo_util.SignalActor.options(name=f'gloo_{self._group_name}_signal', lifetime='detached').remote(self._context.size)
                    else:
                        time.sleep(0.1)
                elapsed = datetime.datetime.now() - start_time
            if not q:
                raise RuntimeError('Unable to get gloo_queue.')
            if self._context.rank == 0:
                ray.get(q.put_nowait.remote(self._group_name))
            while ray.get(q.index.remote(self._group_name)):
                time.sleep(0.1)
            self._context.connectFullMesh(self._store, self._device)
            ray.get(s.send.remote(self._context.rank))
            if self._context.rank == 0:
                ray.get(s.wait.remote())
                keys = []
                keys += [f'rank_{i}' for i in range(self._context.size)]
                keys += [f'{i}' for i in range(self._context.size)]
                self._store.delKeys(keys)
                group_name = ray.get(q.get_nowait.remote())
                assert group_name == self._group_name
                ray.kill(s)

    @property
    def store_type(self):
        if False:
            print('Hello World!')
        return self._store_type

    @property
    def store(self):
        if False:
            while True:
                i = 10
        return self._store

    @property
    def device_type(self):
        if False:
            print('Hello World!')
        return self._device_type

    @property
    def device(self):
        if False:
            while True:
                i = 10
        return self._device

    def destroy(self):
        if False:
            print('Hello World!')
        'GC the store and device used by this rendevzous.'
        self._device = None

class GLOOGroup(BaseGroup):

    def __init__(self, world_size, rank, group_name, store_type='ray_internal_kv', device_type='tcp'):
        if False:
            while True:
                i = 10
        'Init an GLOO collective group.\n\n        Args:\n            world_size: The number of processes.\n            rank: The id of process\n            group_name: The unique user-specified group name.\n            store_type: The store type. Optional: "redis",\n                              "file", "hash".\n            device_type: The device type to transport.\n                               Optional: "tcp", "uv".\n        '
        super(GLOOGroup, self).__init__(world_size, rank, group_name)
        self._gloo_context = gloo_util.create_gloo_context(self.rank, self.world_size)
        self._rendezvous = Rendezvous(self.group_name, self._gloo_context, store_type, device_type)
        self._rendezvous.meet()

    def destroy_group(self):
        if False:
            print('Hello World!')
        'Destroy the group and release GLOO communicators.'
        self._rendezvous.destroy()
        if self._gloo_context is not None:
            pygloo.barrier(self._gloo_context)
            self._gloo_context = None
        if self.rank == 0 and self._rendezvous.store_type == 'file':
            store_name = get_store_name(self._group_name)
            store_path = gloo_util.get_gloo_store_path(store_name)
            if os.path.exists(store_path):
                shutil.rmtree(store_path)
        super(GLOOGroup, self).destroy_group()

    @classmethod
    def backend(cls):
        if False:
            for i in range(10):
                print('nop')
        return Backend.GLOO

    def allreduce(self, tensors, allreduce_options=AllReduceOptions()):
        if False:
            return 10
        'AllReduce a list of tensors following options.\n\n        Args:\n            tensor: the tensor to be reduced, each tensor locates on CPU\n            allreduce_options:\n\n        Returns:\n            None\n        '

        def collective_fn(input_tensor, output_tensor, context):
            if False:
                return 10
            pygloo.allreduce(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor), gloo_util.get_gloo_reduce_op(allreduce_options.reduceOp))
        self._collective(tensors, tensors, collective_fn)

    def barrier(self, barrier_options=BarrierOptions()):
        if False:
            for i in range(10):
                print('nop')
        'Blocks until all processes reach this barrier.\n\n        Args:\n            barrier_options: barrier options.\n\n        Returns:\n            None\n        '
        barrier_tensor = numpy.array([1])
        self.allreduce([barrier_tensor])

    def reduce(self, tensors, reduce_options=ReduceOptions()):
        if False:
            print('Hello World!')
        'Reduce tensors following options.\n\n        Args:\n            tensors: the list of tensors to be reduced,\n                            this list only have one tensor.\n            reduce_options: reduce options.\n\n        Returns:\n            None\n        '
        root_rank = reduce_options.root_rank

        def collective_fn(input_tensor, output_tensor, context):
            if False:
                return 10
            pygloo.reduce(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor), gloo_util.get_gloo_reduce_op(reduce_options.reduceOp), root_rank)
        self._collective(tensors, tensors, collective_fn)

    def broadcast(self, tensors, broadcast_options=BroadcastOptions()):
        if False:
            while True:
                i = 10
        'Broadcast tensors to all other processes following options.\n\n        Args:\n            tensors: tensors to be broadcast or received.\n            broadcast_options: broadcast options.\n\n        Returns:\n            None\n        '
        root_rank = broadcast_options.root_rank

        def collective_fn(input_tensor, output_tensor, context):
            if False:
                i = 10
                return i + 15
            pygloo.broadcast(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor), root_rank)
        self._collective(tensors, tensors, collective_fn)

    def allgather(self, tensor_lists, tensors, allgather_options=AllGatherOptions()):
        if False:
            return 10
        'Allgather tensors on CPU into a list of tensors.\n\n        Args:\n            tensor_lists (List[List[Tensor]]): allgathered tensors.\n            tensors: the list of tensors to allgather across the group.\n                     Each tensor must locate on CPU.\n            allgather_options: allgather options.\n\n        Returns:\n            None\n        '

        def collective_fn(input_tensor, output_tensor, context):
            if False:
                i = 10
                return i + 15
            pygloo.allgather(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), gloo_util.get_tensor_n_elements(input_tensor), gloo_util.get_gloo_tensor_dtype(input_tensor))
        _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists)
        output_flattened = [_flatten_for_scatter_gather(tensor_list, copy=False) for tensor_list in tensor_lists]

        def postprocess_fn():
            if False:
                i = 10
                return i + 15
            for (i, tensor_list) in enumerate(tensor_lists):
                for (j, tensor) in enumerate(tensor_list):
                    gloo_util.copy_tensor(tensor, output_flattened[i][j])
        self._collective(tensors, output_flattened, collective_fn, postprocess_fn=postprocess_fn)

    def reducescatter(self, tensors, tensor_lists, reducescatter_options=ReduceScatterOptions()):
        if False:
            return 10
        'Reduce the scatter a list of tensors across the group.\n\n        Args:\n            tensors: the output tensors (could be unspecified), each\n                            located on CPU.\n            tensor_lists (List[List]): the list of tensors to be reduced then\n                                       scattered.\n            reducescatter_options: reduce-scatter options.\n\n        Returns:\n            None\n        '

        def collective_fn(input_tensor, output_tensor, context):
            if False:
                print('Hello World!')
            size = gloo_util.get_tensor_n_elements(input_tensor)
            world_size = self._gloo_context.size
            pygloo.reduce_scatter(context, gloo_util.get_tensor_ptr(input_tensor), gloo_util.get_tensor_ptr(output_tensor), size, [size // world_size for _ in range(world_size)], gloo_util.get_gloo_tensor_dtype(output_tensor), gloo_util.get_gloo_reduce_op(reducescatter_options.reduceOp))
        _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists)
        input_flattened = [_flatten_for_scatter_gather(tensor_list, copy=False) for tensor_list in tensor_lists]

        def preprocess_fn():
            if False:
                i = 10
                return i + 15
            for (i, tensor_list) in enumerate(tensor_lists):
                for (j, tensor) in enumerate(tensor_list):
                    gloo_util.copy_tensor(input_flattened[i][j], tensor)
        self._collective(input_flattened, tensors, collective_fn, preprocess_fn=preprocess_fn)

    def send(self, tensors, send_options=SendOptions()):
        if False:
            return 10
        'Send a tensor to a destination rank in the group.\n\n        Args:\n            tensors: the tensor to send.\n            send_options: send options.\n\n        Returns:\n            None\n        '

        def p2p_fn(tensor, context, peer):
            if False:
                i = 10
                return i + 15
            pygloo.send(context, gloo_util.get_tensor_ptr(tensor), gloo_util.get_tensor_n_elements(tensor), gloo_util.get_gloo_tensor_dtype(tensor), peer)
        self._point2point(tensors, p2p_fn, send_options.dst_rank)

    def recv(self, tensors, recv_options=RecvOptions()):
        if False:
            for i in range(10):
                print('nop')
        'Receive a tensor from a source rank in the group.\n\n        Args:\n            tensors: the received tensor.\n            recv_options: Receive options.\n\n        Returns:\n            None\n        '

        def p2p_fn(tensor, context, peer):
            if False:
                while True:
                    i = 10
            pygloo.recv(context, gloo_util.get_tensor_ptr(tensor), gloo_util.get_tensor_n_elements(tensor), gloo_util.get_gloo_tensor_dtype(tensor), peer)
        self._point2point(tensors, p2p_fn, recv_options.src_rank)

    def _collective(self, input_tensors, output_tensors, collective_fn, preprocess_fn=None, postprocess_fn=None):
        if False:
            print('Hello World!')
        'A method to encapsulate all collective calls.\n\n        Args:\n            input_tensors: the list of the input tensors.\n            output_tensors: the list of the output tensors.\n            collective_fn: the collective function call.\n            preprocess_fn: preprocess procedures before collective calls.\n            postprocess_fn: postprocess procedures after collective calls.\n\n        Returns:\n            None\n        '
        _check_cpu_tensors(input_tensors)
        _check_cpu_tensors(output_tensors)
        if preprocess_fn:
            preprocess_fn()
        collective_fn(input_tensors[0], output_tensors[0], self._gloo_context)
        if postprocess_fn:
            postprocess_fn()

    def _point2point(self, tensors, p2p_fn, peer_rank: int):
        if False:
            return 10
        'A method to encapsulate all peer-to-peer calls (i.e., send/recv).\n\n        Args:\n            tensors: the tensor to send or receive.\n            p2p_fn: the p2p function call.\n            peer_rank: the rank of the peer process.\n\n        Returns:\n            None\n        '
        _check_cpu_tensors(tensors)
        p2p_fn(tensors[0], self._gloo_context, peer_rank)

def _check_cpu_tensors(tensors):
    if False:
        return 10
    'Check only have one tensor and located on CPU.'
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("'tensors' must be a nonempty list.")
    if len(tensors) != 1:
        raise RuntimeError('Gloo only accept one tensor in the tensor list. Got {} != 1.'.format(len(tensors)))
    d = gloo_util.get_tensor_device(tensors[0])
    if d != 'cpu':
        raise RuntimeError('Gloo only accept cpu tensor . Got {}.'.format(d))

def _flatten_for_scatter_gather(tensor_list, copy=False):
    if False:
        i = 10
        return i + 15
    'Flatten the tensor for gather/scatter operations.\n\n    Args:\n        tensor_list: the list of tensors to be scattered/gathered.\n        copy: whether the copy the tensors in tensor_list into the buffer.\n\n    Returns:\n        The flattened tensor buffer.\n    '
    if not tensor_list:
        raise RuntimeError('Received an empty list.')
    t = tensor_list[0]
    dtype = gloo_util.get_numpy_tensor_dtype(t)
    buffer_shape = [len(tensor_list)] + gloo_util.get_tensor_shape(t)
    buffer = numpy.empty(buffer_shape, dtype=dtype)
    if copy:
        for (i, tensor) in enumerate(tensor_list):
            gloo_util.copy_tensor(buffer[i], tensor)
    return buffer

def _check_inputs_compatibility_for_scatter_gather(tensors, tensor_lists):
    if False:
        i = 10
        return i + 15
    'Check the compatibility between tensor input and tensor list input.'
    if not tensors or not isinstance(tensors, list):
        raise RuntimeError("The first argument 'tensors' expects a list of tensors.")
    if len(tensors) != 1:
        raise RuntimeError("Gloo only accept one tensor in the first argument 'tensors'. Got {} != 1.".format(len(tensors)))
    if not tensor_lists or not isinstance(tensor_lists, list):
        raise RuntimeError("The second argument 'tensor_lists' expects a list of tensor list.")
    if len(tensor_lists) != 1:
        raise RuntimeError("Gloo only accept one tensor list in the second argument 'tensor_lists'. Got {} != 1.".format(len(tensor_lists)))
    dtype = gloo_util.get_gloo_tensor_dtype(tensors[0])
    shape = gloo_util.get_tensor_shape(tensors[0])
    for t in tensor_lists[0]:
        dt = gloo_util.get_gloo_tensor_dtype(t)
        if dt != dtype:
            raise RuntimeError("All tensor operands to scatter/gather must have the same dtype. Got '{}' and '{}'.".format(dt, dtype))
        s = gloo_util.get_tensor_shape(t)
        if s != shape:
            raise RuntimeError("All tensor operands to scatter/gather must have the same shape. Got '{}' and '{}'.".format(s, shape))