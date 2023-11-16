import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.framework import core
from ..meta_parallel.parallel_layers.random import get_rng_state_tracker
from ..meta_parallel.pp_utils import utils
from .recompute import check_recompute_necessary, detach_variable, swith_rng_state_tracker
__all__ = []

def _split_activation(tensor, mp_group):
    if False:
        print('Hello World!')
    mp_degree = mp_group.nranks
    mp_rank = mp_group.rank
    if mp_degree < 2:
        return tensor
    tensor_numel = paddle.numel(tensor)
    assert tensor_numel != 0, "can't recompute zero element"
    assert tensor_numel % mp_degree == 0, 'The capacity of the activation ({}) cannot be divisible by mp_degree({})'.format(tensor_numel, mp_degree)
    data = tensor.flatten_()
    part_size = tensor_numel // mp_degree
    start = part_size * mp_rank
    end = start + part_size
    return data[start:end]

def _merge_activation(tensor, mp_group):
    if False:
        return 10
    mp_degree = mp_group.nranks
    mp_rank = mp_group.rank
    if mp_degree < 2:
        return tensor
    tensor_shape = list(tensor.shape)
    tensor_shape[0] *= mp_group.nranks
    out = paddle.empty(tensor_shape, tensor.dtype)
    task = mp_group.process_group.all_gather(tensor.cuda(), out)
    task.wait()
    return out

class _HPRecomputeFunction(PyLayer):
    """
    Compared with paddle.distributed.fleet.utils.recompute, there are the following differences:
    1. In order to support PipeLineParallel, the input of recompute is modified to ensure that the input can be tuple type.
    2. Offload support for activation
    3. Support MP segmentation of activation to further reduce cuda memory
    4. Adapt to the random state of MP
    """

    @staticmethod
    def forward(ctx, run_function, all_outputs, mp_group, offload, partition, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ctx.run_function = run_function
        ctx.kwargs = kwargs
        ctx.fwd_rng_state = paddle.get_rng_state()
        ctx.fwd_rng_state_tracker = get_rng_state_tracker().get_states_tracker()
        ctx.mp_group = mp_group
        ctx.offload = offload
        ctx.partition = partition
        ctx.inputs = []
        ctx.tensor_indices = []
        ctx.tensor_shapes = []
        tensor_inputs = []
        cur_device = paddle.get_device()
        assert 'gpu:' in paddle.get_device() or 'xpu:' in paddle.get_device() or cur_device.split(':')[0] in paddle.device.get_all_custom_device_type(), f'Recompute with RNG is not support current device: {cur_device}.'
        tracer = framework._dygraph_tracer()
        ctx.is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
        if tracer._amp_level == core.AmpLevel.O2:
            ctx.amp_level = 'O2'
        elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
            ctx.amp_level = 'O1'
        else:
            raise ValueError(f'unsupported amp level: {tracer._amp_level}')
        ctx.amp_dtype = tracer._amp_dtype
        (ctx.amp_white_list, ctx.amp_black_list) = tracer._get_amp_op_list()
        with paddle.no_grad():
            outputs = run_function(*args, **kwargs)
        for (i, arg) in enumerate(args):
            if paddle.is_tensor(arg):
                state = arg.stop_gradient
                if partition:
                    ctx.tensor_shapes.append(arg.shape)
                    partition = _split_activation(arg.detach(), mp_group).clone()
                    arg = partition.cpu() if offload else partition
                else:
                    arg = arg.cpu() if offload else arg
                arg.stop_gradient = state
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
                if framework.in_dynamic_mode() and state:
                    ctx.mark_non_differentiable(arg)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)
        if paddle.is_tensor(outputs):
            all_outputs += [outputs]
            return outputs
        else:
            all_outputs += outputs
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        if False:
            for i in range(10):
                print('nop')
        with paddle.base.dygraph.guard():
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            tensor_shapes = ctx.tensor_shapes
            tensors = list(ctx.saved_tensor())
            device_id = paddle.distributed.ParallelEnv().device_id
            for (i, idx) in enumerate(tensor_indices):
                if ctx.partition:
                    state = tensors[i].stop_gradient
                    tensors[i] = _merge_activation(tensors[i], ctx.mp_group).detach().reshape_(tensor_shapes[i])
                    tensors[i].stop_gradient = state
                inputs[idx] = tensors[i].cuda(device_id) if ctx.offload else tensors[i]
            tracer = framework._dygraph_tracer()
            tracer._has_grad = True
            with swith_rng_state_tracker(ctx.fwd_rng_state, ctx.fwd_rng_state_tracker):
                if ctx.is_fw_autocast:
                    with paddle.amp.auto_cast(enable=ctx.is_fw_autocast, custom_white_list=ctx.amp_white_list, custom_black_list=ctx.amp_black_list, level=ctx.amp_level, dtype=ctx.amp_dtype):
                        detached_inputs = detach_variable(tuple(inputs))
                        outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
                else:
                    detached_inputs = detach_variable(tuple(inputs))
                    outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
            if isinstance(outputs, core.eager.Tensor):
                outputs = (outputs,)
            assert len(outputs) == len(args)
            forward_outputs_with_grad = []
            backward_inputs = []
            for i in range(len(outputs)):
                if isinstance(outputs[i], core.eager.Tensor) and (not outputs[i].stop_gradient):
                    forward_outputs_with_grad.append(outputs[i])
                    backward_inputs.append(args[i])
            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError('none of output has stop_gradient=False, this recompute() is not necessary')
            paddle.autograd.backward(forward_outputs_with_grad, backward_inputs)
            grads = tuple((inp._grad_ivar() for inp in detached_inputs if isinstance(inp, core.eager.Tensor)))
            return grads

def recompute_hybrid(ctx, function, *args, **kwargs):
    if False:
        return 10
    "\n    recompute intermediate activations to save the memory in hybrid parallel scene.\n    # NODTE(shenliang03)The current hybrid parallel recompute has limitations.\n    # It cannot handle the following situations:\n    # 1. The calculation output of recompute, there are tensors that do not require gradients.\n    # 2. The forward output tensor has no gradient. This problem can be solved temporarily by detach().\n    # 3. Here, we only use float dtype to distinguish whether a gradient is needed in output tensor\n\n    Parameters:\n        ctx(dict): include 'mp_group', 'offload', and 'partition' keys. the key 'mp_group' (Group), represents the avtivations are splitted\n                   in which group. the key 'offload' (bool, optional, default=False), represents whether to offload to cpu. the key 'partition' (bool, optional, default=False),\n                   represents whether to split activations in the mp_group.\n        function(paddle.nn.Layer): layer of sequence of layers that describes part of forward pass of the model\n              whose intermediate activations will be released to save memory in forward stage and will be recomputed\n              in backward stage for gradient calculation.\n        *args(Tensor): inputs(tuple) to the function.\n\n        **kwargs(Dict): inputs(dict) to the function.\n\n    Returns:\n        Output of function on args and kwargs.\n\n    "
    mp_group = ctx.get('mp_group', None)
    assert mp_group is not None, 'ctx must contains mp_group and mp_group can not be None.'
    offload = ctx.get('offload', False)
    partition = ctx.get('partition', False)
    if framework._dygraph_tracer()._has_grad:
        check_recompute_necessary(args)
    all_outputs = []
    _HPRecomputeFunction.apply(function, all_outputs, mp_group, offload, partition, *args, **kwargs)
    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        for output in all_outputs:
            if paddle.is_tensor(output) and (not utils.is_float_tensor(output)):
                output.stop_gradient = True
        return tuple(all_outputs)