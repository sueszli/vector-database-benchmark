import contextlib
import copy
import weakref
import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.base.framework import EagerParamBase
from paddle.distributed.fleet.meta_parallel.parallel_layers.random import get_rng_state_tracker
from paddle.framework import core, in_dynamic_mode
from ..utils.log_util import logger
__all__ = []

def _varbase_help(param):
    if False:
        return 10
    state = copy.deepcopy(param.__dict__)
    new_param = EagerParamBase(shape=param.shape, dtype=param.dtype, name=param.name, **state)
    param._share_buffer_to(new_param)
    return new_param

def detach_variable(inputs):
    if False:
        for i in range(10):
            print('nop')
    out = []
    for inp in inputs:
        if not isinstance(inp, core.eager.Tensor) and (type(inp) is not tuple or not isinstance(inp[0], core.eager.Tensor)):
            out.append(inp)
            continue
        if isinstance(inp, EagerParamBase):
            out.append(_varbase_help(inp))
            continue
        if type(inp) is tuple:
            detach_inp = []
            for i in inp:
                assert isinstance(i, core.eager.Tensor)
                if isinstance(i, EagerParamBase):
                    detach_inp.append(_varbase_help(i))
                else:
                    tmp_i = i.detach()
                    tmp_i.stop_gradient = i.stop_gradient
                    detach_inp.append(tmp_i)
            out.append(tuple(detach_inp))
            continue
        x = inp.detach()
        x.stop_gradient = inp.stop_gradient
        out.append(x)
    return tuple(out)

def check_recompute_necessary(inputs):
    if False:
        return 10
    necessary_for_each_input = []
    for input_ in inputs:
        if isinstance(input_, (core.eager.Tensor, paddle.Tensor)):
            necessary_for_each_input.append(input_.stop_gradient)
        elif type(input_) is tuple:
            for i in input_:
                if isinstance(i, (core.eager.Tensor, paddle.Tensor)):
                    necessary_for_each_input.append(i.stop_gradient)
    if all(necessary_for_each_input):
        logger.warning('[Recompute]: None of the inputs to current recompute block need grad, therefore there is NO need to recompute this block in backward !')

@contextlib.contextmanager
def swith_rng_state_tracker(rng_state, tracker):
    if False:
        i = 10
        return i + 15
    orig_rng_state = paddle.get_rng_state()
    orig_rng_tracker = get_rng_state_tracker().get_states_tracker()
    paddle.set_rng_state(rng_state)
    get_rng_state_tracker().set_states_tracker(tracker)
    try:
        yield
    finally:
        paddle.set_rng_state(orig_rng_state)
        get_rng_state_tracker().set_states_tracker(orig_rng_tracker)

class RecomputeFunction(PyLayer):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args, **kwargs):
        if False:
            print('Hello World!')
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.kwargs = kwargs
        ctx.inputs = []
        ctx.tensor_indices = []
        ctx.duplicate_tensor = [False for _ in range(len(args))]
        tensor_inputs = []
        for (i, arg) in enumerate(args):
            if paddle.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            elif type(arg) is tuple:
                is_tensors = [paddle.is_tensor(a) for a in arg]
                if all(is_tensors):
                    tensors_stop_gradient = [a.stop_gradient for a in arg]
                    if not all(tensors_stop_gradient) and any(tensors_stop_gradient):
                        raise ValueError('Recompute receive a tuple containing tensor holds different stop gradient.')
                    tensor_inputs.append(arg)
                    ctx.tensor_indices.append(i)
                    ctx.duplicate_tensor[i] = True
                    ctx.inputs.append(None)
                elif any(is_tensors):
                    raise ValueError('Recompute receive a tuple containing tensor and non-tensor at same time.')
                else:
                    ctx.inputs.append(arg)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)
        if ctx.preserve_rng_state:
            ctx.fw_rng_state = paddle.get_rng_state()
            ctx.fwd_rng_state_tracker = get_rng_state_tracker().get_states_tracker()
        tracer = framework._dygraph_tracer()
        ctx.is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
        if tracer._amp_level == core.AmpLevel.O2:
            ctx.amp_level = 'O2'
        elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
            ctx.amp_level = 'O1'
        else:
            raise ValueError(f'unsupported amp level: {tracer._amp_level}')
        if tracer._amp_dtype == 'float16':
            ctx.amp_dtype = 'float16'
        elif tracer._amp_dtype in ('bfloat16', 'float32'):
            ctx.amp_dtype = 'bfloat16'
        else:
            raise ValueError(f'unsupported amp dtype: {tracer._amp_dtype}')
        (ctx.amp_white_list, ctx.amp_black_list) = tracer._get_amp_op_list()
        with paddle.no_grad():
            outputs = run_function(*args, **kwargs)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if False:
            print('Hello World!')
        with paddle.base.dygraph.guard():
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            duplicate_tensor = ctx.duplicate_tensor
            tensors = ctx.saved_tensor()
            for (i, idx) in enumerate(tensor_indices):
                inputs[idx] = tensors[i]
            tracer = framework._dygraph_tracer()
            tracer._has_grad = True
            if ctx.preserve_rng_state:
                with swith_rng_state_tracker(ctx.fw_rng_state, ctx.fwd_rng_state_tracker):
                    with paddle.amp.auto_cast(enable=ctx.is_fw_autocast, custom_white_list=ctx.amp_white_list, custom_black_list=ctx.amp_black_list, level=ctx.amp_level, dtype=ctx.amp_dtype):
                        detached_inputs = detach_variable(tuple(inputs))
                        outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
            else:
                with paddle.amp.auto_cast(enable=ctx.is_fw_autocast, custom_white_list=ctx.amp_white_list, custom_black_list=ctx.amp_black_list, level=ctx.amp_level, dtype=ctx.amp_dtype):
                    detached_inputs = detach_variable(tuple(inputs))
                    outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
            if isinstance(outputs, core.eager.Tensor):
                outputs = (outputs,)
            assert len(outputs) == len(args)
            forward_outputs_with_grad = []
            backward_inputs_with_grad = []
            for i in range(len(outputs)):
                if isinstance(outputs[i], core.eager.Tensor) and (not outputs[i].stop_gradient):
                    forward_outputs_with_grad.append(outputs[i])
                    backward_inputs_with_grad.append(args[i])
            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError('none of output has requires_grad=True, this recompute() is not necessary')
            with paddle.amp.auto_cast(enable=False):
                paddle.autograd.backward(forward_outputs_with_grad, backward_inputs_with_grad)
            grads = []
            for (idx, inp) in enumerate(detached_inputs):
                if isinstance(inp, core.eager.Tensor):
                    grads.append(inp._grad_ivar())
                elif type(inp) is tuple and duplicate_tensor[idx]:
                    if all((i.stop_gradient for i in inp)):
                        grads.append(None)
                    else:
                        grads.append(tuple((i._grad_ivar() for i in inp)))
            if in_dynamic_mode():
                grads = tuple(grads)
            else:
                grads = list(grads)
            return grads

def _recompute_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    if False:
        return 10
    '\n    recompute without reentrant, that means use hook to implement the recompute function rather than re-entrant autograd.\n    '
    if preserve_rng_state:
        cur_device = paddle.get_device()
        if 'gpu:' in cur_device:
            fw_cuda_rng_state = paddle.get_cuda_rng_state()
        elif 'xpu:' in cur_device:
            fw_cuda_rng_state = paddle.get_rng_state()
        elif cur_device.split(':')[0] in paddle.device.get_all_custom_device_type():
            fw_cuda_rng_state = paddle.get_rng_state(cur_device)
        else:
            raise RuntimeError('Recompute with RNG perserve is not support current device: {}.'.format(cur_device))
        fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states_tracker()
    tracer = framework._dygraph_tracer()
    is_fw_autocast = False if tracer._amp_level == core.AmpLevel.O0 else True
    if tracer._amp_level == core.AmpLevel.O2:
        amp_level = 'O2'
    elif tracer._amp_level in (core.AmpLevel.O1, core.AmpLevel.O0):
        amp_level = 'O1'
    if tracer._amp_dtype == 'float16':
        amp_dtype = 'float16'
    elif tracer._amp_dtype in ('bfloat16', 'float32'):
        amp_dtype = 'bfloat16'
    (amp_white_list, amp_black_list) = tracer._get_amp_op_list()

    class Intermediate_Holder:
        pass
    storage = weakref.WeakKeyDictionary()
    holder_list = []

    def pack(x):
        if False:
            while True:
                i = 10
        res = Intermediate_Holder()
        holder_list.append(weakref.ref(res))
        return res

    def unpack(x):
        if False:
            while True:
                i = 10
        unpack_counter = 0
        if len(storage) == 0:

            def inner_pack(inner_x):
                if False:
                    return 10
                nonlocal unpack_counter
                unpack_counter += 1
                if holder_list[unpack_counter - 1]() is None:
                    return
                tmp_tensor = core.eager.Tensor(inner_x.dtype, inner_x.shape, inner_x.name + 'cpy', core.VarDesc.VarType.LOD_TENSOR, inner_x.persistable)
                inner_x._share_buffer_to(tmp_tensor)
                storage[holder_list[unpack_counter - 1]()] = tmp_tensor
                return

            def inner_unpack(inner_x):
                if False:
                    while True:
                        i = 10
                raise Exception('An unexcepted backward called on a tensor!')
            if preserve_rng_state:
                with swith_rng_state_tracker(fw_cuda_rng_state, fwd_cuda_rng_state_tracker):
                    with paddle.set_grad_enabled(True):
                        with paddle.amp.auto_cast(enable=is_fw_autocast, custom_white_list=amp_white_list, custom_black_list=amp_black_list, level=amp_level, dtype=amp_dtype):
                            with paddle.autograd.saved_tensors_hooks(inner_pack, inner_unpack):
                                unused_outputs = function(*args, **kwargs)
            else:
                with paddle.set_grad_enabled(True), paddle.amp.auto_cast(enable=is_fw_autocast, custom_white_list=amp_white_list, custom_black_list=amp_black_list, level=amp_level, dtype=amp_dtype), paddle.autograd.saved_tensors_hooks(inner_pack, inner_unpack):
                    unused_outputs = function(*args, **kwargs)
        if x not in storage:
            raise Exception('Not supported to retrieve a tensor saved by autograd multiple times that is no need to recompute.')
        return storage[x]
    with paddle.autograd.saved_tensors_hooks(pack, unpack):
        outputs = function(*args, **kwargs)
    return outputs

def recompute(function, *args, **kwargs):
    if False:
        return 10
    '\n    recompute intermediate activations to save then memory.\n\n    Parameters:\n        function(paddle.nn.Layer): layer of sequence of layers that describes part of forward pass of the model\n              whose intermediate activations will be released to save memory in forward stage and will be recomputed\n              in backward stage for gradient calculation.\n        *args(Tensor): inputs to the function.\n        **kwargs(Dict): Kwargs should only contain two kinds of key-value params, the one is part of function\'s key-value params,\n                        and the other contains \'preserve_rng_state\' and \'use_reentrant\'. the key-value pair of preserve_rng_state,\n                        which is used to indicate whether to save the forward rng. If it is True, then the last forward rng value\n                        will be restored when the forward recalculation of backpropagation is performed, its default value is True.\n                        the key-value pair of use_reentrant is used to indicate which implementation of recompute you will be used.\n                        \'use_reentrant=True\' means to use the PyLayer implementation of recompute, \'use_reentrant=False\' means to\n                        use the Hook implementation of recompute, its default value is True.\n    Returns:\n        Output of function on args.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:DISTRIBUTED, env:GPU)\n            >>> import paddle\n            >>> from paddle.distributed.fleet.utils import recompute\n            >>> import random\n            >>> paddle.seed(2023)\n            >>> def get_fc_block(block_idx, input_size, is_last=False):\n            ...     block_name = "block_" + str(block_idx)\n            ...     block = paddle.nn.Sequential(\n            ...         (block_name + "_fc_0", paddle.nn.Linear(input_size, input_size, bias_attr=False)),\n            ...         (block_name + "_dropout", paddle.nn.Dropout(p=0.5)),\n            ...         (block_name + "_relu_1", paddle.nn.ReLU()),\n            ...         (block_name + "_fc_1", paddle.nn.Linear(input_size, input_size, bias_attr=False)),\n            ...         (block_name + "_relu_2", paddle.nn.ReLU()),\n            ...     )\n            ...     if is_last:\n            ...         block.add_sublayer(\n            ...             block_name + "_fc_2",\n            ...             paddle.nn.Linear(\n            ...                 input_size, 1, bias_attr=False\n            ...             )\n            ...         )\n            ...     else:\n            ...         block.add_sublayer(\n            ...             block_name + "_fc_2",\n            ...             paddle.nn.Linear(input_size, input_size, bias_attr=False)\n            ...         )\n            ...     return block\n\n            >>> class Naive_fc_net(paddle.nn.Layer):\n            ...     def __init__(self, input_size=10,\n            ...                 recompute_blocks=[1, 3],\n            ...                 recompute_kwargs={}):\n            ...         super().__init__()\n            ...         self.recompute_blocks = recompute_blocks\n            ...         self.recompute_kwargs = recompute_kwargs\n            ...         self.runfunc0 = get_fc_block(0, input_size, is_last=False)\n            ...         self.runfunc1 = get_fc_block(1, input_size, is_last=False)\n            ...         self.runfunc2 = get_fc_block(2, input_size, is_last=False)\n            ...         self.runfunc3 = get_fc_block(3, input_size, is_last=False)\n            ...         self.runfunc4 = get_fc_block(4, input_size, is_last=True)\n            ...         self.total_func = [self.runfunc0, self.runfunc1, self.runfunc2, self.runfunc3, self.runfunc4]\n            ...     def forward(self, inputs):\n            ...         nums = len(self.total_func)\n            ...         for i in range(nums):\n            ...             if i in self.recompute_blocks:\n            ...                 inputs = recompute(self.total_func[i], inputs, **{"preserve_rng_state": True})\n            ...             else:\n            ...                 inputs = self.total_func[i](inputs)\n            ...         return inputs\n\n            >>> def run_model(cuda_state, recompute_block=[], recompute_kwargs={}):\n            ...     gen = paddle.seed(10)\n            ...     gen.manual_seed(10)\n            ...     random.seed(10)\n            ...     if cuda_state:\n            ...         paddle.set_cuda_rng_state(cuda_state)\n            ...     batch_size, input_size = 1, 10\n            ...     model = Naive_fc_net(\n            ...         input_size,\n            ...         recompute_blocks=recompute_block,\n            ...         recompute_kwargs=recompute_kwargs)\n            ...     optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())\n            ...     loss_ = []\n            ...     param_ = []\n            ...     grad_ = []\n            ...     for _ in range(5):\n            ...         x = paddle.rand(shape=[batch_size, input_size], dtype="float32")\n            ...         y_pred = model(x)\n            ...         loss = y_pred.mean()\n            ...         loss_.append(loss.item())\n            ...         loss.backward()\n            ...         optimizer.step()\n            ...         param_.append(model.parameters()[9])\n            ...         grad_.append(model.parameters()[3]._grad_ivar())\n            ...         optimizer.clear_grad()\n            ...     return loss_, param_, grad_\n\n            >>> cuda_state = paddle.get_cuda_rng_state()\n            >>> # without recompute\n            >>> loss_ref, param_ref, grad_ref = run_model(\n            ...     cuda_state, recompute_block=[]\n            ... )\n\n            >>> loss, param, grad = run_model(cuda_state, recompute_block=[1, 2])\n            >>> print("normal_loss: {}, recompute_loss: {}".format(loss_ref, loss))\n            >>> # The result of the recompute_loss should be the same as the normal_loss.\n            normal_loss: [0.0018744759727269411, 0.0, 0.035971127450466156, 0.0, 0.0], recompute_loss: [0.0018744759727269411, 0.0, 0.035971127450466156, 0.0, 0.0]\n\n    '
    preserve = kwargs.pop('preserve_rng_state', True)
    use_reentrant = kwargs.pop('use_reentrant', True)
    if kwargs and use_reentrant:
        raise ValueError('Error, if you want to send kwargs(dict parameter) to function, please set use_reentrant=False.')
    if framework._dygraph_tracer()._has_grad:
        check_recompute_necessary(args)
    if use_reentrant:
        return RecomputeFunction.apply(function, preserve, *args)
    else:
        return _recompute_without_reentrant(function, preserve, *args, **kwargs)

def recompute_sequential(ctx, functions, *args, **kwargs):
    if False:
        return 10
    "\n    recompute intermediate activations to save the memory for 'Sequential' models. use 'ctx' to transmit some context params, it is similar to 'recompute_hybrid' API.\n\n    Parameters:\n        ctx(dict): include 'segments' and  'preserve_rng_state' keys, the key 'segments' (int, default 1), represents the number of chunks to create in the model,\n                   the key 'preserve_rng_state' (bool, optional, default=True) indicate whether to save the forward rng. If it is True, then the last forward rng value will be\n                   restored when the forward recalculation of backpropagation is performed.\n        functions(paddle.nn.Sequential): layer of sequence of layers that describes part of forward pass of the model\n              whose intermediate activations will be released to save memory in forward stage and will be recomputed\n              in backward stage for gradient calculation.\n        *args(Tensor): inputs(tuple) to the function.\n        **kwargs(Dict): inputs(dict) to the function.\n\n    Returns:\n        Output of function on args and kwargs.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n            >>> import paddle\n            >>> from paddle.incubate.distributed.fleet import recompute_sequential\n            >>> input = paddle.ones(shape=[8, 10])\n            >>> model = paddle.nn.Sequential(paddle.nn.Linear(10, 10), paddle.nn.Linear(10, 2))\n            >>> output = recompute_sequential({'segments' : 1}, model, input)\n\n    "
    segments = ctx.get('segments', 1)
    preserve_rng_state = ctx.get('preserve_rng_state', True)

    def _run_func(begin, end, funcs):
        if False:
            for i in range(10):
                print('nop')

        def do_run(input):
            if False:
                i = 10
                return i + 15
            for i in range(begin, end + 1):
                input = funcs[i](input)
            return input
        return do_run
    if isinstance(functions, paddle.nn.Sequential):
        functions = list(functions.children())
    segment_size = len(functions) // segments
    end = -1
    for begin in range(0, segment_size * (segments - 1), segment_size):
        end = begin + segment_size - 1
        args = recompute(_run_func(begin, end, functions), *args, preserve_rng_state=preserve_rng_state, **kwargs)
    return _run_func(end + 1, len(functions) - 1, functions)(args)