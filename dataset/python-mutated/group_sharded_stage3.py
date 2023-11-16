import logging
from collections import OrderedDict
from types import MethodType
import numpy as np
import paddle
import paddle.distributed as dist
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.base.framework import EagerParamBase
from paddle.distributed import collective
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm
from .group_sharded_storage import GradStorage
from .group_sharded_utils import GroupShardedClipGrad, Type, device_guard

def _all_gather(tensor, buffer_size, group):
    if False:
        return 10
    '\n    The main difference with paddle.distributed.all_gather:\n    no need to pass in tensor_list, the returned tensor is spliced\n    '
    assert group is not None
    if framework.in_dynamic_mode():
        out = paddle.zeros([buffer_size], dtype=tensor.dtype)
        task = group.process_group.all_gather(tensor, out)
        return (out, task)
alignment = {'gpu': 256, 'cpu': 4096, 'xpu': 256}
align = {Type.bf16.value: 2, Type.fp16.value: 2, Type.fp32.value: 4}
global CHECK_LAYER
CHECK_LAYER = {}

class GroupShardedStage3(nn.Layer):
    """
    A wrapper for Sharding Stage3 Layer in Dygraph.

    .. warning: GroupShardedStage3 encapsulates the layer strategy and integrates it into the nn.Layer.

    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf.
    """

    def __init__(self, layer, optimizer, group=None, sync_buffers=False, device='gpu', segment_size=2 ** 20, pertrain_sync_models=True, offload=False, sync_comm=False, dp_group=None, exclude_layer=None):
        if False:
            return 10
        super().__init__()
        assert core.is_compiled_with_cuda() or device in core.get_all_custom_device_type(), 'Only support CUDA / CustomDevice.'
        self._layer = layer
        self._default_device = device
        self.__sync_buffers = sync_buffers
        self._offload = offload
        self._sync_comm = sync_comm
        self._exclude_layer = [] if exclude_layer is None else exclude_layer
        assert isinstance(self._exclude_layer, (list, tuple)), "the exclude_layers must be a list with layers' name or layers' id"
        assert segment_size >= 0, 'segment_size must be GE than 0.'
        self._segment_size = segment_size
        global DEV
        DEV = 'cpu' if paddle.get_device() == 'cpu' else paddle.get_device().split(':')[0]
        global DEV_ID
        DEV_ID = 0 if paddle.get_device() == 'cpu' else int(paddle.get_device().split(':')[1])
        global param2dtype
        param2dtype = {}
        self._group = collective.new_group(collective._get_global_group().ranks) if group is None else group
        self._dp_group = dp_group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert self._group.nranks > 1, 'Training must be distributed, ranks must be greater than 1.'
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[0]
        self._param2buffer_size = {}
        self._param2buffer = {}
        self._trainable_params = {}
        self._unslice_params = set()
        self._unslice_params2align = {}
        self._grad_storages = {}
        assert not isinstance(optimizer, list), 'Multiple optimizers are not supported now.'
        self._optim = _OptimizerWrapper(optimizer, self._offload, self._group, self._update_params_slice)
        self._ori_parameter_list = self._optim._parameter_list
        self._ori_param_groups = self._optim._param_groups
        if isinstance(self._optim._grad_clip, ClipGradByGlobalNorm):
            logging.warning('While using ClipGradByGlobalNorm in GroupShardedStage3, the grad clip of original optimizer will be changed.')
            self._optim._grad_clip = GroupShardedClipGrad(self._optim._grad_clip, paddle.get_device(), self._group)
            if self._optim._parameter_list and isinstance(self._optim._parameter_list[0], dict):
                for item in self._optim._param_groups:
                    if 'grad_clip' in item.keys():
                        item['grad_clip'] = self._optim._grad_clip
        self._check_main_grad()
        if pertrain_sync_models:
            self._sync_params_and_buffers()
        self._segment_rank_params(self._layer)
        self._handle_unslice_params()
        self._order_tracer = OrderedDict()
        self._order_tracer['order'] = 0
        self._order_tracer['layer'] = []
        self._task_flow = TaskFlow()
        self._register_forward_hooks(self._layer)
        self._register_backward_hooks()
        self._redefine_opt_step()
        self._redefine_opt_clear()

    def _check_main_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_main_grad = None
        for param in self._layer.parameters():
            if self.use_main_grad is None and hasattr(param, 'main_grad'):
                self.use_main_grad = True
            if self.use_main_grad:
                assert hasattr(param, 'main_grad'), 'Params have different main grad attributes.'

    @paddle.autograd.no_grad()
    def _sync_params_and_buffers(self):
        if False:
            print('Hello World!')
        '\n        Sync all model states for all ranks\n        '
        for p in self._layer.parameters():
            dist.broadcast(p, src=self._global_root_rank, group=self._group, sync_op=True)
            if self._dp_group is not None and self._dp_group.nranks > 1:
                dist.broadcast(p, src=self._dp_group.ranks[0], group=self._dp_group, sync_op=True)

    def _clear_gradients(self):
        if False:
            while True:
                i = 10
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(filter(lambda p: p.trainable and p not in self._unslice_params, current_layer_params))
        for param in trainable_params:
            assert hasattr(param, 'fw_storage'), f"Find {param.name} don't have fw_storage attribute."
            if self.use_main_grad:
                param.fw_storage.main_grad._clear()
                param.fw_storage.main_grad = None
            else:
                param.fw_storage.clear_gradient(False)
            param.bw_storage._clear()
            param.bw_storage = None
        if not self._offload:
            for grad_storage in self._grad_storages.values():
                grad_storage.buffer.zero_()
        else:
            for param in list(self._unslice_params):
                if self.use_main_grad:
                    param.main_grad._clear()
                    param.main_grad = None
                else:
                    param.clear_gradient(False)
                if self._default_device in paddle.device.get_all_custom_device_type():
                    tmp_var = param._copy_to(paddle.CustomPlace(self._default_device, DEV_ID), True)
                else:
                    tmp_var = param.cuda(DEV_ID)
                if tmp_var.dtype == Type.fp32.value and param2dtype[param.name] == Type.fp16.value:
                    tmp_var = paddle.cast(tmp_var, Type.fp16.value)
                elif tmp_var.dtype == Type.fp32.value and param2dtype[param.name] == Type.bf16.value:
                    tmp_var = paddle.cast(tmp_var, Type.bf16.value)
                tmp_var._share_buffer_to(param)
                del tmp_var
            for grad_storage in self._grad_storages.values():
                grad_storage.manumal_relase()
                grad_storage.rebuild()

    def _update_params_slice(self):
        if False:
            print('Hello World!')
        update_list = self._update_params()
        if not isinstance(self._optim._param_groups[0], dict):
            slice_params = [param.fw_storage for param in update_list]
            self._optim._parameter_list = slice_params + list(self._unslice_params)
            self._optim._param_groups = slice_params + list(self._unslice_params)
        else:
            for param_group in self._optim._param_groups:
                p_group = []
                for p in param_group['params']:
                    if hasattr(p, 'fw_storage'):
                        p_group.append(p.fw_storage)
                    else:
                        p_group.append(p)
                param_group['params'] = p_group

    def forward(self, *inputs, **kwargs):
        if False:
            print('Hello World!')
        '\n        A wrapper for Sharding Stage3 layer.\n        '
        if self.__sync_buffers:
            self._sync_buffers()
        fw = self._layer(*inputs, **kwargs)
        return fw

    def set_state_dict(self, state_dict, use_structured_name=True):
        if False:
            while True:
                i = 10
        self._layer.set_state_dict(state_dict, use_structured_name=use_structured_name)

    def state_dict(self, destination=None, include_sublayers=True, structured_name_prefix=''):
        if False:
            return 10
        return self._layer.state_dict(destination=destination, include_sublayers=include_sublayers, structured_name_prefix=structured_name_prefix)

    def _handle_unslice_params(self):
        if False:
            i = 10
            return i + 15
        buffer_size = {}
        buffer_size[Type.bf16.value] = 0
        buffer_size[Type.fp32.value] = 0
        buffer_size[Type.fp16.value] = 0
        for param in self._unslice_params:
            if (param.dtype == Type.fp16.value or param.dtype == Type.bf16.value) and (not self._offload):
                master_tensor = paddle.cast(param, Type.fp32.value)
                master_tensor.name = param.name
                self._optim._master_weights[param.name] = master_tensor
            if self._offload:
                param.master_weight = paddle.cast(param, Type.fp32.value).cpu()
            param2dtype[param.name] = param.dtype
            p_align = self._param2align(param)
            self._unslice_params2align[param.name] = p_align
            buffer_size[param.dtype] += param._numel() + p_align
        for param in sorted(self._unslice_params, key=lambda p: p.name):
            if param.dtype not in self._grad_storages.keys():
                self._grad_storages[param.dtype] = GradStorage(buffer_size[param.dtype], dtype=param.dtype if not self.use_main_grad else paddle.float32, device=self._default_device, destination=self._rank, parm2align=self._unslice_params2align)
            self._grad_storages[param.dtype].add_grad(param, self._unslice_params2align[param.name])

    def _segment_rank_params(self, layer, name='last_layer'):
        if False:
            i = 10
            return i + 15
        '\n        Flatten parameters according to layer.\n        '
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            CHECK_LAYER[id(layer)] = name
            self._flatten_layer_params(layer, current_layer_params)
        for (name, sub_layer) in layer.named_children():
            self._segment_rank_params(sub_layer, name)

    def _flatten_layer_params(self, layer, current_layer_params):
        if False:
            print('Hello World!')
        '\n        Parameter segmentation and memory integration.\n        '
        if id(layer) in self._trainable_params.keys():
            return
        if id(layer) in self._exclude_layer or layer.__class__.__name__ in self._exclude_layer:
            for p in current_layer_params:
                if p.trainable:
                    self._unslice_params.add(_UnsliceParam(p))
            return

        def _add_manage_info(trainable_param):
            if False:
                i = 10
                return i + 15
            return _PartitionParam(trainable_param)
        current_params = []
        for p in current_layer_params:
            if p._numel() > self._segment_size:
                current_params.append(_add_manage_info(p))
            elif p.trainable:
                self._unslice_params.add(_UnsliceParam(p))
        self._trainable_params[id(layer)] = current_params
        for param in self._trainable_params[id(layer)]:
            if param.name in self._param2buffer.keys():
                continue
            self._param2buffer[param.name] = []
            align_ = self._param2align(param)
            offset = align_ + param._numel()
            buffer_size = offset if offset % self._group.nranks == 0 else offset + self._group.nranks - offset % self._group.nranks
            self._param2buffer_size[param.name] = buffer_size
            assert buffer_size % self._group.nranks == 0
            pre_buffer = buffer_size // self._group.nranks
            for rank_ in range(self._group.nranks):
                self._param2buffer[param.name].append((rank_ * pre_buffer, (rank_ + 1) * pre_buffer))
            param2dtype[param.name] = param.dtype
            self._param_storage(param, buffer_size)

    def _param_storage(self, param, buffer_size):
        if False:
            while True:
                i = 10
        '\n        This is a function to simplify the handling of parameter InternalStorages.\n        '
        assert isinstance(buffer_size, int)
        value = np.zeros(buffer_size, dtype=np.float16) if Type.fp16.value == param.dtype or Type.bf16.value == param.dtype else np.zeros(buffer_size, dtype=np.float32)
        buffer = core.eager.Tensor(value=value, place=core.CPUPlace())
        if Type.bf16.value == param.dtype:
            buffer = buffer.cast(Type.bf16.value)
        param_shape = param.shape
        origin_state = param.stop_gradient
        param.stop_gradient = True
        param.flatten_()
        param.stop_gradient = origin_state
        (start, end) = self._param2buffer[param.name][self._rank]
        with device_guard():
            tmp_var = buffer._slice(0, param._numel())
        param_cpu = param.cpu()
        tmp_var.get_tensor().set(param_cpu.get_tensor(), core.CPUPlace())
        del tmp_var
        param.get_tensor()._set_dims(param_shape)
        if self._offload:
            with device_guard():
                tmp_tensor = buffer._slice(start, end)
            param.fw_storage = core.eager.Tensor(value=tmp_tensor, place=core.CPUPlace(), name='slice@' + param.name)
            if param.trainable:
                with device_guard():
                    param.master_weight = paddle.cast(param.fw_storage, Type.fp32.value)
        else:
            param.fw_storage = core.eager.Tensor(value=buffer._slice(start, end), name='slice@' + param.name)
        param.status = 'part'
        if param.trainable and (param.dtype == Type.fp16.value or param.dtype == Type.bf16.value) and (not self._offload):
            master_tensor = paddle.cast(param.fw_storage, Type.fp32.value)
            master_tensor.name = param.name
            self._optim._master_weights[param.fw_storage.name] = master_tensor
        param._clear_data()

    def _register_forward_hooks(self, layer):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register PyLayer to manage memory slices.\n        There are four stages:\n        FW\n        1. Before the forward layers, synchronize the full parameters.\n        2. After the forward layers, release the full parameter and keep the parameter slice.\n        BW\n        3. Before the backward layers, synchronize the full parameters and create param's grad.\n        4. After the gradient accumulation, release the full parameter and keep the parameter slice.\n        "
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            if not (id(layer) in self._exclude_layer or layer.__class__.__name__ in self._exclude_layer):
                self._register_forward_all_hooks(layer, self._task_flow)
        for (_, sub_layer) in layer.named_children():
            self._register_forward_hooks(sub_layer)

    def _register_forward_all_hooks(self, sub_layer, task_flow):
        if False:
            print('Hello World!')

        def _forward_pre_hook(layer, inputs):
            if False:
                while True:
                    i = 10
            return ForwardPreHooks(layer, self._order_tracer, self._trainable_params, self._param2buffer_size, self._group, self._sync_comm, self._offload, task_flow)

        def _forward_post_hook(layer, inputs, outputs):
            if False:
                i = 10
                return i + 15
            return ForwardPostHooks.apply(outputs, layer, self._order_tracer, self._trainable_params, self._param2buffer, self._param2buffer_size, self._rank, self._group, self._sync_comm, self._offload, task_flow)
        sub_layer.register_forward_pre_hook(_forward_pre_hook)
        sub_layer.register_forward_post_hook(_forward_post_hook)

    @paddle.autograd.no_grad()
    def _sync_buffers(self):
        if False:
            return 10
        '\n        Sync all the param buffers from all ranks (exp: batch norm statistics).\n        '
        for buffer in self._layer.buffers(include_sublayers=True):
            dist.broadcast(buffer, self._global_root_rank, self._group, sync_op=True)
            if self._dp_group is not None and self._dp_group.nranks > 1:
                dist.broadcast(buffer, self._dp_group.ranks[0], self._dp_group, sync_op=True)

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        'Forward missing attributes to wrapped layer.'
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)

    def _update_params(self):
        if False:
            print('Hello World!')
        '\n        Update parameters to optimizer memory slice.\n        '
        update_list = []
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(filter(lambda p: p.trainable and p not in self._unslice_params, current_layer_params))
        for param in trainable_params:
            assert hasattr(param, 'fw_storage'), f"Find {param.name} don't have fw_storage attribute"
            param.fw_storage = _TensorWrapper(param)
            if self.use_main_grad:
                param.fw_storage.main_grad = param.bw_storage
            else:
                assert param.fw_storage.grad is None
                param.fw_storage._copy_gradient_from(param.bw_storage)
            update_list.append(param)
        for grad_storage in self._grad_storages.values():
            grad_storage.buffer.scale_(scale=self._world_size_scaling)
            dist.all_reduce(tensor=grad_storage.buffer, group=self._group)
            if self._dp_group is not None and self._dp_group.nranks > 1:
                grad_storage.buffer.scale_(scale=1.0 / self._dp_group.nranks)
                dist.all_reduce(tensor=grad_storage.buffer, group=self._dp_group)
        if self._offload:
            for param in list(self._unslice_params):
                param._clear_data()
                param.master_weight._share_buffer_to(param)
            for grad_storage in self._grad_storages.values():
                for p in grad_storage._params:
                    if self.use_main_grad:
                        tmp_g = _device2cpu(p.main_grad, convert_dtype=True)
                        p.main_grad = tmp_g
                    else:
                        tmp_g = _device2cpu(p.grad, convert_dtype=True)
                        p.clear_gradient(False)
                        p._copy_gradient_from(tmp_g)
                    del tmp_g
                grad_storage.buffer._clear()
        return update_list

    def get_all_parameters(self, convert2cpu=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the full parameters and return the corresponding task flows.\n        '
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(filter(lambda p: p.trainable and p not in self._unslice_params, current_layer_params))
        t_flow = _allgather_buffer(trainable_params, self._group, param2buffer_size=self._param2buffer_size, use_calc_stream=True, task_flow=TaskFlow(), sync_wait=True, offload=self._offload, convert2cpu=convert2cpu)
        if convert2cpu:
            for param in trainable_params:
                t_flow.full_param[param.name][0]._share_buffer_to(param)
                del t_flow.full_param[param.name]
        self._optim._parameter_list = self._ori_parameter_list
        self._optim._param_groups = self._ori_param_groups

    def _register_backward_hooks(self):
        if False:
            print('Hello World!')
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(filter(lambda p: p.trainable and p not in self._unslice_params, current_layer_params))
        for param in trainable_params:
            allreduce_function = self._get_allreduce_fn(param)
            param._register_backward_hook(allreduce_function)

    def _get_allreduce_fn(self, param):
        if False:
            print('Hello World!')

        @paddle.autograd.no_grad()
        def allreduce_(*_):
            if False:
                for i in range(10):
                    print('nop')
            assert param.trainable, 'the param must be trainable for grad allreduced'
            if param.name in self._task_flow.full_grad.keys():
                full_grad = self._task_flow.full_grad[param.name]
                full_grad.scale_(scale=self._world_size_scaling)
                dist.all_reduce(tensor=full_grad, group=self._group)
                if self._dp_group is not None and self._dp_group.nranks > 1:
                    full_grad.scale_(scale=1.0 / self._dp_group.nranks)
                    dist.all_reduce(tensor=full_grad, group=self._dp_group)
                (start, end) = self._param2buffer[param.name][self._rank]
                if param.bw_storage is None:
                    param.bw_storage = full_grad._slice(start, end).detach().clone()
                    if self._offload:
                        param.bw_storage = _device2cpu(param.bw_storage, True)
                elif self._offload:
                    cpu_grad = _device2cpu(full_grad._slice(start, end).detach().clone(), True)
                    with device_guard():
                        param.bw_storage = paddle.add(param.bw_storage, cpu_grad)
                else:
                    param.bw_storage = paddle.add(param.bw_storage, full_grad._slice(start, end).detach().clone())
                if self.use_main_grad:
                    param.main_grad = None
                else:
                    param.clear_gradient(False)
                del self._task_flow.full_grad[param.name]
            if param.name in self._task_flow.full_param.keys():
                if param.status == 'all':
                    param.use_count = 0
                    param._clear_data()
                    (start, end) = self._param2buffer[param.name][self._rank]
                    param.fw_storage = self._task_flow.full_param[param.name][0]._slice(start, end).detach().clone()
                    param.status = 'part'
                    del self._task_flow.full_param[param.name]
                    if self._offload:
                        param.fw_storage._clear_data()
                        param.master_weight._share_buffer_to(param.fw_storage)
        return allreduce_

    def _param2align(self, param):
        if False:
            return 10
        size = param._numel() * align[param.dtype]
        if self._default_device in core.get_all_custom_device_type():
            device_alignment = core.libpaddle._get_device_min_chunk_size(self._default_device)
        else:
            device_alignment = alignment[self._default_device]
        remaining = size % device_alignment
        ali = 0 if remaining == 0 else device_alignment - remaining
        align_ = ali // align[param.dtype]
        return align_

    def _redefine_opt_step(self):
        if False:
            while True:
                i = 10
        params_slice_func = self._update_params_slice
        opt_step = self._optim.step

        def _opt_step(self):
            if False:
                while True:
                    i = 10
            if not self.update_scaler:
                params_slice_func()
            if self.offload:
                with device_guard():
                    opt_step()
            else:
                opt_step()

        def _opt_minimize(self):
            if False:
                i = 10
                return i + 15
            raise RuntimeError('optimizer.minimize() not support now, please use optimizer.step()')
        self._optim.step = MethodType(_opt_step, self._optim)
        self._optim.minimize = MethodType(_opt_minimize, self._optim)

    def _redefine_opt_clear(self):
        if False:
            i = 10
            return i + 15
        clear_func = self._clear_gradients

        def _opt_clear(self):
            if False:
                return 10
            clear_func()
        self._optim.clear_grad = MethodType(_opt_clear, self._optim)

def ForwardPreHooks(layer, order_tracer, trainable_params, param2buffer_size, group, sync_comm, offload, task_flow):
    if False:
        while True:
            i = 10
    layer_id = id(layer)
    (use_calc, sync_wait) = (False, False)
    if layer_id not in order_tracer.keys() or sync_comm:
        (use_calc, sync_wait) = (True, True)
        task_flow.use_calc[layer_id] = use_calc
    else:
        task_flow.use_calc[layer_id] = use_calc
        _wait_layer(trainable_params[layer_id], task_flow, group, param2buffer_size, use_calc, offload)
        if layer_id == order_tracer['layer'][-1]:
            return
        order_ = order_tracer[layer_id]
        layer_id = order_tracer['layer'][order_ + 1]
    _allgather_buffer(trainable_params[layer_id], group, param2buffer_size=param2buffer_size, use_calc_stream=use_calc, task_flow=task_flow, sync_wait=sync_wait, offload=offload)
    return

class ForwardPostHooks(PyLayer):

    @staticmethod
    def forward(ctx, inputs, layer, order_tracer, trainable_params, param2buffer, param2buffer_size, rank, group, sync_comm, offload, task_flow):
        if False:
            for i in range(10):
                print('nop')
        layer_id = id(layer)
        _release_param(trainable_params[layer_id], param2buffer, rank, task_flow, offload)
        if layer_id not in order_tracer.keys():
            order_ = order_tracer['order']
            order_tracer[layer_id] = order_
            order_tracer['order'] += 1
            order_tracer['layer'].append(layer_id)
        ctx.order_tracer = order_tracer
        ctx.task_flow = task_flow
        ctx.group = group
        ctx.layer_id = layer_id
        ctx.sync_comm = sync_comm
        ctx.trainable_params = trainable_params
        ctx.param2buffer_size = param2buffer_size
        ctx.offload = offload
        return inputs

    @staticmethod
    def backward(ctx, *args):
        if False:
            print('Hello World!')
        order_tracer = ctx.order_tracer
        task_flow = ctx.task_flow
        group = ctx.group
        layer_id = ctx.layer_id
        trainable_params = ctx.trainable_params
        param2buffer_size = ctx.param2buffer_size
        sync_comm = ctx.sync_comm
        offload = ctx.offload
        (use_calc, sync_wait) = (False, False)
        if sync_comm:
            (use_calc, sync_wait) = (True, True)
            _allgather_buffer(trainable_params[layer_id], group, param2buffer_size=param2buffer_size, use_calc_stream=use_calc, task_flow=task_flow, sync_wait=sync_wait, offload=offload)
        else:
            _wait_layer(trainable_params[layer_id], task_flow, group, param2buffer_size, use_calc, offload)
        _create_params_grad(trainable_params[layer_id], param2buffer_size, task_flow)
        task_flow.use_calc[layer_id] = use_calc
        if layer_id != order_tracer['layer'][0] and (not sync_comm):
            layer_next_id = order_tracer['layer'][order_tracer[layer_id] - 1]
            _allgather_buffer(trainable_params[layer_next_id], group, param2buffer_size=param2buffer_size, use_calc_stream=use_calc, task_flow=task_flow, sync_wait=sync_wait, offload=offload)
        return args

class TaskFlow:
    """
    Task flows, one way linked list for task acquisition.
    """

    def __init__(self, full_param={}, full_grad={}, use_calc={}, callback=None):
        if False:
            return 10
        self.full_param = full_param
        self.full_grad = full_grad
        self.use_calc = use_calc
        self.callback = callback

def _release_param(trainable_params, param2buffer, rank, task_flow, offload=False):
    if False:
        for i in range(10):
            print('nop')
    for param in trainable_params:
        param.use_count -= 1
        if param.use_count == 0:
            param._clear_data()
            if param.name in task_flow.full_param.keys():
                (start, end) = param2buffer[param.name][rank]
                with paddle.amp.auto_cast(enable=False):
                    param.fw_storage = task_flow.full_param[param.name][0]._slice(start, end).detach().clone()
                param.status = 'part'
                del task_flow.full_param[param.name]
                if offload:
                    param.fw_storage = _device2cpu(param.fw_storage)

def _wait_layer(trainable_params, task_flow, group, param2buffer_size, use_calc_stream, offload=False):
    if False:
        print('Hello World!')
    for param in trainable_params:
        if param.status == 'all':
            param.use_count += 1
            continue
        if param.name in task_flow.full_param.keys():
            (full_param, task) = task_flow.full_param[param.name]
            task.wait()
            full_param._slice(0, param._numel())._share_buffer_to(param)
            param.fw_storage._clear()
            param.fw_storage = None
            param.status = 'all'
            param.use_count += 1
        else:
            _allgather_buffer(trainable_params, group, param2buffer_size=param2buffer_size, use_calc_stream=True, task_flow=task_flow, sync_wait=True, offload=offload)
            break
    return task_flow

def _allgather_buffer(trainable_params, group, param2buffer_size, use_calc_stream, task_flow, sync_wait=False, offload=False, convert2cpu=False):
    if False:
        for i in range(10):
            print('nop')
    if convert2cpu:
        assert sync_wait
    for param in trainable_params:
        if param.status == 'all':
            param.use_count += 1
            continue
        if offload:
            param.fw_storage = _cpu2device(param)
        buffer_size = param2buffer_size[param.name]
        with paddle.amp.auto_cast(enable=False):
            (full_param, task) = _all_gather(param.fw_storage, buffer_size, group)
        if sync_wait:
            with paddle.amp.auto_cast(enable=False):
                task.wait()
            if convert2cpu:
                cpu_full_param = _device2cpu(full_param._slice(0, param._numel()))
                full_param._clear_data()
                del full_param
                full_param = cpu_full_param
                task = None
            else:
                full_param._slice(0, param._numel())._share_buffer_to(param)
                param.fw_storage._clear()
                param.fw_storage = None
                param.status = 'all'
                param.use_count += 1
        task_flow.full_param[param.name] = (full_param, task)
    return task_flow

@paddle.autograd.no_grad()
def _create_params_grad(trainable_params, param2buffer_size, task_flow):
    if False:
        print('Hello World!')
    for param in trainable_params:
        use_main_grad = hasattr(param, 'main_grad')
        if not param.trainable:
            continue
        if param.name in task_flow.full_grad.keys():
            continue
        assert isinstance(param2buffer_size[param.name], int)
        temp_grad = paddle.zeros([param2buffer_size[param.name]], dtype=param.dtype if not use_main_grad else paddle.float32)
        temp_tensor = temp_grad._slice(0, param._numel())
        temp_tensor.get_tensor()._set_dims(param.shape)
        if use_main_grad:
            param.main_grad = temp_tensor
        else:
            param._copy_gradient_from(temp_tensor)
        del temp_tensor
        task_flow.full_grad[param.name] = temp_grad
    return task_flow

def _PartitionParam(param):
    if False:
        print('Hello World!')
    if not hasattr(param, 'fw_storage'):
        param.fw_storage = None
        param.bw_storage = None
        param.master_weight = None
        param.status = 'all'
        param.use_count = 0
    return param

def _UnsliceParam(param):
    if False:
        while True:
            i = 10
    if not hasattr(param, 'unslice'):
        param.unslice = True
        param.master_weight = None
    return param

def _TensorWrapper(param):
    if False:
        print('Hello World!')
    var = param.fw_storage
    tmp_param = EagerParamBase(shape=var.shape, dtype=var.dtype, name='slice@' + param.name)
    var._share_buffer_to(tmp_param)
    tmp_param.regularizer = param.regularizer
    tmp_param.optimize_attr['learning_rate'] = param.optimize_attr['learning_rate']
    var._clear()
    return tmp_param

def _OptimizerWrapper(optimizer, offload, group, update_params_slice):
    if False:
        i = 10
        return i + 15
    if not hasattr(optimizer, '_optim'):
        optimizer._optim = optimizer
        optimizer.offload = offload
        optimizer._group = group
        optimizer.update_scaler = None
        optimizer.update_slice = update_params_slice
    return optimizer

def _device2cpu(trans_param, convert_dtype=False):
    if False:
        return 10
    if convert_dtype:
        trans_param = paddle.cast(trans_param, Type.fp32.value)
    tmp_p = trans_param.cpu()
    trans_param._clear_data()
    return tmp_p

def _cpu2device(param):
    if False:
        print('Hello World!')
    if DEV in paddle.device.get_all_custom_device_type():
        tmp_p = param.fw_storage._copy_to(paddle.CustomPlace(DEV, DEV_ID), True)
    else:
        tmp_p = param.fw_storage.cuda(DEV_ID)
    if tmp_p.dtype == Type.fp32.value and param2dtype[param.name] == Type.fp16.value:
        tmp_p = paddle.cast(tmp_p, Type.fp16.value)
    elif tmp_p.dtype == Type.fp32.value and param2dtype[param.name] == Type.bf16.value:
        tmp_p = paddle.cast(tmp_p, Type.bf16.value)
    return tmp_p

def _current_layer_params(layer):
    if False:
        while True:
            i = 10
    return layer.parameters(include_sublayers=False) + list(layer.extra_parameters) if hasattr(layer, 'extra_parameters') else layer.parameters(include_sublayers=False)