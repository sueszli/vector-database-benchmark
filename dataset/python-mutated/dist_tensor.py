import copy
import inspect
import paddle
from paddle.framework import Block
from paddle.static import Parameter, Variable
from .dist_attribute import TensorDistAttr
from .utils import __no_shape_var_type__, _linear_idx2coordinate

class DistributedTensor:
    """
    DistributedTensor represents the distribution of tensor on the process group and
    local tensors can be created by DistributedTensor.
    Only support even sharding now and uneven sharding will be supported in the future.
    Local tensor information can be obtained from the DistributedTensor instance object,
    or obtained by the static methods provided by DistributedTensor,
    including shard (i.e. the index in the serial tensor), offsets, and sizes.
    """

    @staticmethod
    def _validate_sizes_and_dist_attr(sizes, dims_mapping, topology, processes, rank=None, shard_sizes=None):
        if False:
            print('Hello World!')
        if not (isinstance(sizes, (list, tuple)) and all((isinstance(x, int) and x >= 0 for x in sizes))):
            raise ValueError(f'The sizes must be list or tuple and item in sizes must be non-negative integer, but got {sizes}')
        if not (isinstance(dims_mapping, (list, tuple)) and all((isinstance(x, int) and x >= -1 for x in dims_mapping))):
            raise ValueError('The dims_mapping must be list or tuple and item in dims_mapping must >= -1, but got {}'.format(dims_mapping))
        if not (isinstance(processes, (list, tuple)) and all((isinstance(x, int) and x >= 0 for x in processes))):
            raise ValueError('The processes must be list or tuple and item in processes must be integer, but got {}'.format(processes))
        if not (isinstance(topology, (list, tuple)) and all((isinstance(x, int) and x > 0 for x in topology))):
            raise ValueError('The topology must be list or tuple and item in topology must be non-negative integer, but got {}'.format(topology))
        if rank is not None and (not (isinstance(rank, int) and rank >= 0)):
            raise ValueError(f'The rank must >= 0, but got {rank}')

    @staticmethod
    def get_local_sizes(global_sizes, dims_mapping, topology, processes, rank=None, shard_sizes=None):
        if False:
            while True:
                i = 10
        DistributedTensor._validate_sizes_and_dist_attr(global_sizes, dims_mapping, topology, processes, rank, shard_sizes)
        local_sizes = []
        for (idx, item) in enumerate(global_sizes):
            val = dims_mapping[idx] if idx < len(dims_mapping) else -1
            if val == -1:
                local_sizes.append(item)
            else:
                local_sizes.append(item // topology[dims_mapping[idx]])
        return local_sizes

    @staticmethod
    def get_local_offsets(global_sizes, dims_mapping, topology, processes, rank, shard_sizes=None):
        if False:
            print('Hello World!')
        local_sizes = DistributedTensor.get_local_sizes(global_sizes, dims_mapping, topology, processes, rank, shard_sizes)
        local_offsets = []
        rank_relatvie = processes.index(rank)
        coordinate = _linear_idx2coordinate(topology, rank_relatvie)
        for i in range(len(global_sizes)):
            if dims_mapping[i] == -1:
                local_offsets.append(0)
            else:
                local_offsets.append(coordinate[dims_mapping[i]] * local_sizes[i])
        return local_offsets

    @staticmethod
    def get_global_sizes(local_sizes, dims_mapping, topology, processes, rank=None, shard_sizes=None):
        if False:
            print('Hello World!')
        DistributedTensor._validate_sizes_and_dist_attr(local_sizes, dims_mapping, topology, processes, rank, shard_sizes)
        global_sizes = []
        for (idx, item) in enumerate(local_sizes):
            if dims_mapping[idx] == -1:
                global_sizes.append(item)
            else:
                global_sizes.append(item * topology[dims_mapping[idx]])
        return global_sizes

    @staticmethod
    def get_local_shard(global_sizes, dims_mapping, topology, processes, rank, shard_sizes=None):
        if False:
            print('Hello World!')
        local_offsets = DistributedTensor.get_local_offsets(global_sizes, dims_mapping, topology, processes, rank, shard_sizes)
        local_sizes = DistributedTensor.get_local_sizes(global_sizes, dims_mapping, topology, processes, rank, shard_sizes)
        assert len(local_sizes) == len(local_offsets), 'The length of local_sizes must be equal to local_offsets, but got {} and {}.'.format(len(local_sizes), len(local_offsets))
        local_end_offsets = [x[0] + x[1] for x in zip(local_offsets, local_sizes)]
        local_shard = list(zip(local_offsets, local_end_offsets))
        return local_shard

    def __init__(self, serial_tensor, dist_attr=None, dist_context=None):
        if False:
            i = 10
            return i + 15
        self._serial_tensor = serial_tensor
        if dist_attr is not None and isinstance(dist_attr, TensorDistAttr):
            self._dist_attr = copy.deepcopy(dist_attr)
            self._serial_tensor.dist_attr = dist_attr
        else:
            assert dist_attr is None, f'{dist_attr}'
            self._dist_attr = self._serial_tensor.dist_attr
        self._batch_dim = 0
        self._local_offsets_map = {}
        self._local_shard_map = {}
        self._local_tensor_map = {}
        from .dist_context import get_default_distributed_context
        self._dist_context = dist_context if dist_context is not None else get_default_distributed_context()

    @property
    def serial_tensor(self):
        if False:
            print('Hello World!')
        return self._serial_tensor

    @property
    def dist_attr(self):
        if False:
            while True:
                i = 10
        return self._dist_attr

    @dist_attr.setter
    def dist_attr(self, dist_attr):
        if False:
            i = 10
            return i + 15
        self._dist_attr = dist_attr
        self._serial_tensor.dist_attr = dist_attr

    @property
    def dist_context(self):
        if False:
            i = 10
            return i + 15
        return self._dist_context

    def validate_dist_attr(self):
        if False:
            print('Hello World!')
        if self.serial_tensor.type in __no_shape_var_type__:
            return True
        tensor_shape = self.serial_tensor.shape
        if len(tensor_shape) != len(self.dist_attr.dims_mapping):
            return False
        for i in range(len(self.dist_attr.dims_mapping)):
            if self.dist_attr.dims_mapping[i] < -1 or self.dist_attr.dims_mapping[i] >= len(self.dist_attr.process_mesh.shape):
                return False
        for i in range(len(self.dist_attr.process_mesh.shape)):
            if self.dist_attr.dims_mapping.count(i) > 1:
                return False
        return True

    def local_sizes(self, rank=None):
        if False:
            i = 10
            return i + 15
        'Get local sizes of the given rank.'
        rank = paddle.distributed.get_rank() if rank is None else rank
        global_sizes = self.serial_tensor.shape
        dims_mapping = self.dist_attr.dims_mapping
        processes = self.dist_attr.process_mesh.process_ids
        topology = self.dist_attr.process_mesh.shape
        local_sizes = DistributedTensor.get_local_sizes(global_sizes, dims_mapping, topology, processes, rank)
        return local_sizes

    def local_offsets(self, rank=None):
        if False:
            print('Hello World!')
        rank = paddle.distributed.get_rank() if rank is None else rank
        local_offsets = None
        if rank in self._local_offsets_map.keys():
            local_offsets = self._local_offsets_map[rank]
        else:
            global_sizes = self.serial_tensor.shape
            dims_mapping = self.dist_attr.dims_mapping
            processes = self.dist_attr.process_mesh.process_ids
            topology = self.dist_attr.process_mesh.shape
            local_offsets = DistributedTensor.get_local_offsets(global_sizes, dims_mapping, topology, processes, rank)
            self._local_offsets_map[rank] = local_offsets
        return local_offsets

    def global_sizes(self):
        if False:
            print('Hello World!')
        return self.serial_tensor.shape

    def local_shard(self, rank=None):
        if False:
            for i in range(10):
                print('nop')
        rank = paddle.distributed.get_rank() if rank is None else rank
        local_shard = None
        if rank in self._local_shard_map.keys():
            local_shard = self._local_shard_map[rank]
        else:
            global_sizes = self.serial_tensor.shape
            dims_mapping = self.dist_attr.dims_mapping
            processes = self.dist_attr.process_mesh.process_ids
            topology = self.dist_attr.process_mesh.shape
            local_shard = DistributedTensor.get_local_shard(global_sizes, dims_mapping, topology, processes, rank)
            self._local_shard_map[rank] = local_shard
        return local_shard

    def new_local_tensor(self, block=None, rank=None, name=None):
        if False:
            return 10
        '\n        Create a new local tensor of serial tensor corresponding to rank.\n        Args:\n            block (Block): The block contains the new tensor. Default value is recommend and it will be created in the block of dist main program corresponding to the serial tensor block id. Default: None.\n            rank (int): The rank id. Default value is recommend and it will be the current rank. Default: None.\n        '

        def _copy_kwargs(serial_tensor):
            if False:
                while True:
                    i = 10
            kwargs = {}
            no_need_copy_args = ['self', 'block', 'shape', 'name']
            arg_spec = inspect.getfullargspec(Variable.__init__)
            for key in arg_spec.args:
                if key in no_need_copy_args:
                    continue
                elif key not in kwargs:
                    if key == 'type':
                        kwargs[key] = serial_tensor.desc.type()
                    elif key == 'dtype':
                        kwargs[key] = serial_tensor.desc.dtype()
                    elif key == 'lod_level':
                        kwargs[key] = serial_tensor.desc.lod_level()
                    elif key == 'persistable':
                        kwargs[key] = serial_tensor.desc.persistable()
                    elif key == 'stop_gradient':
                        kwargs[key] = serial_tensor.desc.stop_gradient()
                    elif key == 'need_check_feed':
                        kwargs[key] = serial_tensor.desc.need_check_feed()
                    elif key == 'capacity':
                        continue
                    else:
                        kwargs[key] = self.serial_tensor.__dict__[key]
            if isinstance(serial_tensor, Parameter):
                kwargs['trainable'] = serial_tensor.trainable
                kwargs['optimize_attr'] = serial_tensor.trainable
                kwargs['regularizer'] = serial_tensor.regularizer
                kwargs['do_model_average'] = serial_tensor.do_model_average
                kwargs['need_clip'] = serial_tensor.need_clip
                kwargs['is_distributed'] = serial_tensor.is_distributed
                kwargs['is_parameter'] = serial_tensor.is_parameter
            return kwargs
        if rank is not None and (not (isinstance(rank, int) and rank >= 0)):
            raise ValueError(f'The rank must >= 0, but got {rank}')
        if block is not None and (not isinstance(block, Block)):
            raise TypeError(f'The block must be Block, but got {type(block)}.')
        rank = paddle.distributed.get_rank() if rank is None else rank
        if block is None:
            block_id = self.serial_tensor.block.idx
            block = self.dist_context.dist_main_programs[rank].block(block_id)
        kwargs = _copy_kwargs(self.serial_tensor)
        kwargs['name'] = name
        kwargs['shape'] = self.local_sizes(rank)
        if isinstance(self.serial_tensor, Parameter):
            kwargs.pop('persistable')
            local_tensor = Parameter(block=block, **kwargs)
        else:
            local_tensor = block.create_var(**kwargs)
        local_tensor.desc.set_original_id(self.serial_tensor.desc.id())
        self._local_tensor_map[rank] = local_tensor
        return local_tensor

    def local_tensor(self, rank=None):
        if False:
            while True:
                i = 10
        rank = paddle.distributed.get_rank() if rank is None else rank
        assert rank in self._local_tensor_map, f'The rank {rank} local tensor has not been created.'
        return self._local_tensor_map[rank]

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for (k, v) in self.__dict__.items():
            if k == '_serial_tensor' or k == '_local_tensor_map':
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        str = '{{tensor name: {}, tensor id: {}, tensor original_id {}'.format(self.serial_tensor.desc.name(), self.serial_tensor.desc.id(), self.serial_tensor.desc.original_id())
        if self.dist_attr.is_annotated('process_mesh'):
            annotated_str = 'annotated'
        else:
            annotated_str = 'non-annotated'
        str += f', process_mesh ({annotated_str}): {self.dist_attr.process_mesh}'
        str += f', is_parameter: {self.serial_tensor.is_parameter}'
        if self.dist_attr.is_annotated('dims_mapping'):
            annotated_str = 'annotated'
        else:
            annotated_str = 'non-annotated'
        str += f', dims_mapping ({annotated_str}): {self.dist_attr.dims_mapping} }}'
        return str