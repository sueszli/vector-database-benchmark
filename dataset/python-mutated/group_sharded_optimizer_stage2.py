import logging
import warnings
from collections import OrderedDict
import paddle
import paddle.distributed as dist
from paddle.distributed import ParallelMode, fleet
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm
from paddle.optimizer import Optimizer
HybridParallelClipGrad = fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer.HybridParallelClipGrad
from paddle.distributed.collective import _get_global_group, new_group
from .group_sharded_storage import GradStorage, ParamStorage
from .group_sharded_utils import GroupShardedClipGrad, Type, device_guard
alignment = {'gpu': 256, 'cpu': 4096, 'xpu': 256}
align = {Type.fp16.value: 2, Type.bf16.value: 2, Type.fp32.value: 4}

class GroupShardedOptimizerStage2(Optimizer):
    """
    A wrapper for Sharding Stage2 Optimizer in Dygraph.

    .. warning: ShardingOptimizer encapsulates the optimization strategy and integrates it into the optimizer.

    .. ZeRO: 1.https://arxiv.org/pdf/1910.02054.pdf 2.https://arxiv.org/pdf/1910.02054.pdf.

    """

    def __init__(self, params, optim, group=None, offload=False, device='gpu', pertrain_sync_models=True, dp_group=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(learning_rate=optim._learning_rate, parameters=params)
        assert core.is_compiled_with_cuda() or core.is_compiled_with_xpu() or device in core.get_all_custom_device_type(), 'Only GPU and XPU and CustomDevice is supported now'
        self._dtype_rank_params = OrderedDict()
        self._param2rank = {}
        self.__segment_params = []
        self._rank_buffer_size = {}
        self._param2align = {}
        self._optim = optim
        self._reduce_overlap = False
        self._comm_task = None
        assert hasattr(self._optim, '_master_weights'), 'Must use optimizer with _master_weights attribute'
        self._local_params = []
        if isinstance(params[0], dict):
            for param_group in params:
                self._local_params.extend(list(param_group['params']))
        else:
            self._local_params.extend(list(params))
        self.use_main_grad = None
        for param in self._local_params:
            if self.use_main_grad is None and hasattr(param, 'main_grad'):
                self.use_main_grad = True
            if self.use_main_grad:
                assert hasattr(param, 'main_grad'), 'Params have different main grad attributes.'
        if self.use_main_grad:
            assert not offload, 'offload not support main_grad for now'
        self._default_device = device
        self._pfp16 = len(list(filter(lambda x: x.trainable and x.dtype == Type.fp16.value, self._local_params))) > 0
        self._pbf16 = len(list(filter(lambda x: x.trainable and x.dtype == Type.bf16.value, self._local_params))) > 0
        self._broadcast_overlap = False
        self._forward_pre_hook_remove_helper = []
        try:
            self._broadcast_order_params = sorted(self.local_params, key=lambda x: int(x.name.split('.')[0].split('_')[-1]))
        except ValueError:
            self._broadcast_order_params = None
        self._group = new_group(_get_global_group().ranks) if group is None else group
        self._dp_group = dp_group
        self.world_size = self._group.nranks
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[0]
        if self._dp_group is not None and self._dp_group.nranks > 1:
            assert not offload, 'Not support! when using offload with sharding stage2, please use pure sharding stage2, exclude data parallel.'
        if pertrain_sync_models:
            self._sync_params_and_buffers()
        self.param_storages = {}
        if isinstance(self._optim._grad_clip, ClipGradByGlobalNorm):
            logging.warning('While using ClipGradByGlobalNorm in GroupShardedOptimizerStage2, the grad clip of original optimizer will be changed.')
            hcg = fleet.fleet._hcg if hasattr(fleet.fleet, '_hcg') else None
            if hcg and hcg.get_parallel_mode() is not ParallelMode.DATA_PARALLEL and (not offload):
                self._optim._grad_clip = HybridParallelClipGrad(self._optim._grad_clip, hcg)
            else:
                self._optim._grad_clip = GroupShardedClipGrad(self._optim._grad_clip, paddle.get_device(), self._group)
            if self._optim._parameter_list and isinstance(self._optim._parameter_list[0], dict):
                for item in self._optim._param_groups:
                    if 'grad_clip' in item.keys():
                        item['grad_clip'] = self._optim._grad_clip
        if offload:
            assert self._pfp16, "Only support offload strategy while using 'Adam', 'AdamW' and 'Momentum' optimizer with AMP/Pure FP16"
        self.offload = offload
        self.offload_device = 'cpu'
        self.offload_buffer_size = 0
        self.offload_param2align = {}
        self.offload_params = None
        self.offload_grads = None
        self.dev_id = int(paddle.get_device().split(':')[1])
        self._master_params = {}
        self._update_opt_status()

    def _set_auxiliary_var(self, key, val):
        if False:
            return 10
        super()._set_auxiliary_var(key, val)
        self._optim._set_auxiliary_var(key, val)

    @paddle.autograd.no_grad()
    def _sync_params_and_buffers(self):
        if False:
            return 10
        '\n        Sync all model states for all ranks\n        '
        for p in self._local_params:
            dist.broadcast(p, src=self._global_root_rank, group=self._group, sync_op=True)
            if self._dp_group:
                dist.broadcast(p, src=self._dp_group.ranks[0], group=self._dp_group, sync_op=True)

    def _update_task(self, task):
        if False:
            print('Hello World!')
        if self._reduce_overlap:
            assert task is not None
        self._comm_task = task

    def _set_reduce_overlap(self, reduce_overlap):
        if False:
            for i in range(10):
                print('nop')
        self._reduce_overlap = reduce_overlap

    def _set_broadcast_overlap(self, broadcast_overlap, layers=None, num_groups=None):
        if False:
            print('Hello World!')
        self._broadcast_overlap = broadcast_overlap
        if self._broadcast_overlap:
            assert layers is not None, 'To enable broadcast overlap forward, please pass the module to the function.'
            self._layers = layers
            warnings.warn('Setting overlap broadcast means the `paddle.device.cuda.synchronize()` must be called manually before calling `paddle.save()` and before and inference.')
            if self._broadcast_order_params is None:
                warnings.warn("The param name passed to the optimizer doesn't follow .+_[0-9]+\\..+ patter, overlap broadcast may harm the performance.")
                self._broadcast_order_params = self._local_params
        if num_groups is None or num_groups > len(self._broadcast_order_params):
            warnings.warn('The num_groups for broadcast is larger than the number of params to be broadcast. It will set to default value: 1 (use the default sharding group).')
            num_groups = 1
        assert isinstance(num_groups, int) and num_groups > 0, 'num_groups should be a positive integer'
        self._number_of_broadcast_groups = num_groups
        self._broadcast_groups = [None for _ in range(self._number_of_broadcast_groups)]
        self._broadcast_groups[0] = self._group
        ranks = self._group.ranks
        for i in range(1, self._number_of_broadcast_groups):
            self._broadcast_groups[i] = new_group(ranks)

    def _generate_master_params(self, trainable_params):
        if False:
            while True:
                i = 10
        if self.offload:
            for param in trainable_params:
                if param.name not in self._master_params.keys():
                    self._master_params[param.name] = core.eager.Tensor(name=param.name, value=param.cast(dtype=Type.fp32.value).numpy(), place=core.CPUPlace(), stop_gradient=param.stop_gradient)
        else:
            for param in trainable_params:
                if param.dtype == Type.fp16.value or param.dtype == Type.bf16.value:
                    master_tensor = paddle.cast(param, Type.fp32.value)
                    master_tensor.name = param.name
                    self._optim._master_weights[param.name] = master_tensor

    def _update_opt_status(self):
        if False:
            i = 10
            return i + 15
        'Update optimizer status and parameter storage information, and special functions to be developed.'
        self._integration_params()

    def _segment_params(self):
        if False:
            return 10
        '\n        Divide all optimizer parameters equally into rank.\n        '
        if len(self.__segment_params) == 0:
            (self.__segment_params, param_lists) = ([[] for _ in range(self.world_size)], [[] for _ in range(self.world_size)])
            sizes = [0] * self.world_size
            for param in self._local_params:
                rank = sizes.index(min(sizes))
                param_lists[rank].append(param)
                sizes[rank] += param._numel() if param.trainable else 0
            for (rank, params) in enumerate(param_lists):
                self.__segment_params[rank].extend(params)
        return self.__segment_params

    @property
    def local_params(self):
        if False:
            while True:
                i = 10
        return self._local_params

    @property
    def param2rank(self):
        if False:
            print('Hello World!')
        'Map the params to the rank which owns them'
        if len(self._param2rank) == 0:
            for (rank, params) in enumerate(self._segment_params()):
                for param in params:
                    self._param2rank[param.name] = rank
        return self._param2rank

    @property
    def dtype_rank_params(self):
        if False:
            while True:
                i = 10
        '\n        Divide the parameters into groups according to rank and dtype.\n        '
        if len(self._dtype_rank_params) == 0:
            trainable_params = list(filter(lambda x: x.trainable, self._local_params))
            for param in trainable_params:
                if param.dtype not in self._dtype_rank_params.keys():
                    self._dtype_rank_params[param.dtype] = [[] for _ in range(self.world_size)]
                self._dtype_rank_params[param.dtype][self.param2rank[param.name]].append(param)
            for dtype in self._dtype_rank_params.keys():
                for rank_params in self._dtype_rank_params[dtype]:
                    rank_params.sort(key=lambda x: x._numel())
        return self._dtype_rank_params

    @property
    def rank_buffer_size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Count the memory size of the parameters corresponding to rank under the corresponding dtype.\n        '
        if self._default_device in core.get_all_custom_device_type():
            device_alignment = core.libpaddle._get_device_min_chunk_size(self._default_device)
        else:
            device_alignment = alignment[self._default_device]
        if len(self._rank_buffer_size) == 0:
            for dtype in self.dtype_rank_params.keys():
                if dtype not in self._rank_buffer_size.keys():
                    self._rank_buffer_size[dtype] = {}
                for (dst_rank, per_rank_params) in enumerate(self.dtype_rank_params[dtype]):
                    if dst_rank not in self._rank_buffer_size[dtype].keys():
                        self._rank_buffer_size[dtype][dst_rank] = 0
                    for param in per_rank_params:
                        if not param.trainable:
                            continue
                        size = param._numel() * align[dtype]
                        remaining = size % device_alignment
                        ali = 0 if remaining == 0 else device_alignment - remaining
                        align_ = ali // align[dtype]
                        self._rank_buffer_size[dtype][dst_rank] += param._numel() + align_
                        self._param2align[param.name] = align_
        return self._rank_buffer_size

    def _integration_params(self):
        if False:
            print('Hello World!')
        '\n        Integrate the parameters into a continuous memory according to rank, and support the update of training parameters.\n        '
        for (dtype, per_rank_params) in self.dtype_rank_params.items():
            if dtype not in self.param_storages.keys():
                self.param_storages[dtype] = {}
            for (dst_rank, params) in enumerate(per_rank_params):
                if len(params) > 0:
                    trainable_params = list(filter(lambda x: x.trainable, params))
                    if (self._pfp16 or self._pbf16) and dst_rank == self._rank:
                        self._generate_master_params(trainable_params)
                    if trainable_params:
                        param_storage = ParamStorage(size=self.rank_buffer_size[dtype][dst_rank], dtype=dtype, device=self._default_device)
                        param_storage.add_rank_params(trainable_params, self._param2align)
                        self.param_storages[dtype][dst_rank] = param_storage
        dtype_in_use = list(self.dtype_rank_params.keys())
        dtype_to_pop = list(filter(lambda x: x not in dtype_in_use, self.param_storages.keys()))
        for d in dtype_to_pop:
            self.param_storages.pop(d)
        if self.offload:
            self._optim._master_weights = self._master_params
            cpu_master_params = list(self._master_params.values())
            if self._default_device in core.get_all_custom_device_type():
                device_alignment = core.libpaddle._get_device_min_chunk_size(self._default_device)
            else:
                device_alignment = alignment[self._default_device]
            for param in cpu_master_params:
                size = param._numel() * align[Type.fp32.value]
                remaining = size % device_alignment
                ali = 0 if remaining == 0 else device_alignment - remaining
                align_ = ali // align[Type.fp32.value]
                self.offload_buffer_size += param._numel() + align_
                self.offload_param2align[param.name] = align_
            if cpu_master_params:
                with device_guard(self._rank, self.offload_device):
                    self.offload_params = ParamStorage(size=self.offload_buffer_size, dtype=Type.fp32.value, device=self.offload_device)
                    self.offload_params.buffer.name = 'offload_buffer'
                    self.offload_params.add_rank_params(cpu_master_params, self.offload_param2align, False)
                    self.offload_params.buffer.stop_gradient = False
                    self.offload_grads = GradStorage(size=self.offload_buffer_size, dtype=Type.fp32.value, device=self.offload_device, destination=self._rank, parm2align=self.offload_param2align, convert_cpu=True)
                    for p in cpu_master_params:
                        self.offload_grads.add_grad(p, self.offload_param2align[p.name])
                    self._optim._master_weights[self.offload_params.buffer.name] = self.offload_params.buffer

    def _offload_acc_grad(self, param_name, grad_fp32_cpu):
        if False:
            print('Hello World!')
        'accumulate grads with offload strategy'
        with device_guard(self._rank, self.offload_device):
            if param_name in self._master_params.keys():
                if self._master_params[param_name].grad is None:
                    self._master_params[param_name]._copy_gradient_from(grad_fp32_cpu)
                else:
                    self._master_params[param_name].grad.add_(grad_fp32_cpu)
        self.offload_params.buffer._copy_gradient_from(self.offload_grads.buffer)

    def _offload_scale_grad(self, scale_size):
        if False:
            print('Hello World!')
        'scale grads with offload strategy'
        with device_guard(self._rank, self.offload_device):
            self.offload_grads.buffer.scale_(scale=scale_size)

    def _offload_clear_grad(self):
        if False:
            while True:
                i = 10
        'clear grads with offload strategy'
        with device_guard(self._rank, self.offload_device):
            self.offload_grads.buffer.zero_()

    def _step(self):
        if False:
            return 10
        if self._broadcast_overlap:
            for hook_remove in self._forward_pre_hook_remove_helper:
                hook_remove.remove()
            self._forward_pre_hook_remove_helper = []
        if self.offload:
            params_list = [self.offload_params.buffer]
            if not isinstance(self._optim._param_groups[0], dict):
                self._optim._parameter_list = params_list
                self._optim._param_groups = params_list
        if self.offload:
            with device_guard(device=self.offload_device):
                self._optim.step()
            for param in self._local_params:
                if param.name in self._master_params.keys():
                    if self._default_device in core.get_all_custom_device_type():
                        param.set_value(self._master_params[param.name]._copy_to(paddle.CustomPlace(self._default_device, self.dev_id), True).cast(dtype=param.dtype))
                    else:
                        param.set_value(self._master_params[param.name].cuda(self.dev_id).cast(dtype=param.dtype))
        else:
            self._optim.step()
        self._broadcast_params()

    def step(self):
        if False:
            print('Hello World!')
        "\n        A wrapper for Optimizer's step function to finish the update operation of the optimizer.\n        "
        self._step()

    def minimize(self):
        if False:
            return 10
        raise RuntimeError('optimizer.minimize() not support now, please use optimizer.step()')

    def set_state_dict(self, state_dict):
        if False:
            return 10
        self._optim.set_state_dict(state_dict)

    def state_dict(self):
        if False:
            return 10
        return self._optim.state_dict()

    def _clear_cache(self):
        if False:
            i = 10
            return i + 15
        self.__segment_params.clear()
        self._dtype_rank_params.clear()
        self._param2rank.clear()

    @paddle.autograd.no_grad()
    def _broadcast_params(self):
        if False:
            return 10
        'Broadcast the parameters of the current rank to each rank'
        if self._broadcast_overlap:
            self._broadcast_params_overlap_forward()
        else:
            for dtype_per_rank in self.param_storages.values():
                for (dst_rank, internal_storage) in dtype_per_rank.items():
                    dist.broadcast(tensor=internal_storage.buffer, src=self._group.ranks[dst_rank], group=self._group, sync_op=True)

    def _forward_pre_hook_function(self, tasks):
        if False:
            print('Hello World!')

        def __impl__(x, y):
            if False:
                while True:
                    i = 10
            for task in tasks:
                task.wait()
        return __impl__

    def set_lr(self, lr):
        if False:
            while True:
                i = 10
        super().set_lr(lr)
        self._optim.set_lr(lr)

    def get_lr(self):
        if False:
            return 10
        return self._optim.get_lr()

    @paddle.autograd.no_grad()
    def _broadcast_params_overlap_forward(self):
        if False:
            print('Hello World!')
        group_idx = 0
        param2task = {}
        for x in self._broadcast_order_params:
            if x.trainable:
                group = self._broadcast_groups[group_idx]
                group_idx = (group_idx + 1) % self._number_of_broadcast_groups
                task = dist.broadcast(tensor=x, src=group.ranks[self._param2rank[x.name]], group=group, sync_op=False)
                assert x.name not in param2task
                param2task[x.name] = task
        for layer in self._layers.sublayers():
            if len(layer.sublayers()) == 0:
                tasks = []
                for param in layer.parameters():
                    if param.trainable:
                        if param.name in param2task:
                            tasks.append(param2task[param.name])
                self._forward_pre_hook_remove_helper.append(layer.register_forward_pre_hook(self._forward_pre_hook_function(tasks)))