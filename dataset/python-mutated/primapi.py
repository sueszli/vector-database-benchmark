import logging
import typing
import paddle
from paddle.base import backward, core, framework
from paddle.base.core import prim_config
from paddle.incubate.autograd import primx, utils

@framework.static_only
def forward_grad(outputs, inputs, grad_inputs=None):
    if False:
        return 10
    "Forward mode of automatic differentiation.\n\n    Note:\n        **ONLY available in the static graph mode and primitive operators.**\n\n    Args:\n        outputs(Tensor|Sequence[Tensor]): The output tensor or tensors.\n        inputs(Tensor|Sequence[Tensor]): The input tensor or tensors.\n        grad_inputs(Tensor|Sequence[Tensor]): Optional, the gradient Tensor or\n            Tensors of inputs which has the same shape with inputs, Defaults to\n            None, in this case is equivalent to all ones.\n\n    Returns:\n        grad_outputs(Tensor|Sequence[Tensor]): The gradients for outputs.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import numpy as np\n            >>> import paddle\n\n            >>> paddle.enable_static()\n            >>> paddle.incubate.autograd.enable_prim()\n\n            >>> startup_program = paddle.static.Program()\n            >>> main_program = paddle.static.Program()\n\n            >>> with paddle.static.program_guard(main_program, startup_program):\n            ...     x = paddle.static.data('x', shape=[1], dtype='float32')\n            ...     y = x * x\n            ...     y_grad = paddle.incubate.autograd.forward_grad(y, x)\n            ...     paddle.incubate.autograd.prim2orig()\n            ...\n            >>> exe = paddle.static.Executor()\n            >>> exe.run(startup_program)\n            >>> y_grad = exe.run(main_program, feed={'x': np.array([2.]).astype('float32')}, fetch_list=[y_grad])\n            >>> print(y_grad)\n            [array([4.], dtype=float32)]\n\n            >>> paddle.incubate.autograd.disable_prim()\n            >>> paddle.disable_static()\n    "
    if not utils.prim_enabled():
        raise RuntimeError('forward_grad must be running on primitiveoperators, use enable_prim to turn it on.')
    if not isinstance(outputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected outputs is Tensor|Sequence[Tesnor], but got {type(outputs)}.')
    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], but got {type(inputs)}.')
    (ys, xs, xs_dot) = (utils.as_tensors(outputs), utils.as_tensors(inputs), utils.as_tensors(grad_inputs))
    block = framework.default_main_program().current_block()
    if any((x.block != block for x in xs + ys)):
        raise RuntimeError('Variable in inputs and targets should exist in current block of main program.')
    primx.orig2prim(block)
    ad = primx.Transform(ys[0].block)
    (_, ys_dot) = ad.linearize(xs, ys, xs_dot)
    return ys_dot[0] if isinstance(outputs, framework.Variable) else ys_dot

@framework.static_only
def grad(outputs, inputs, grad_outputs=None):
    if False:
        i = 10
        return i + 15
    "Reverse mode of automatic differentiation.\n\n    Note:\n        **ONLY available in the static graph mode and primitive operators**\n\n    Args:\n        outputs(Tensor|Sequence[Tensor]): The output Tensor or Tensors.\n        inputs(Tensor|Sequence[Tensor]): The input Tensor or Tensors.\n        grad_outputs(Tensor|Sequence[Tensor]): Optional, the gradient Tensor or\n            Tensors of outputs which has the same shape with outputs, Defaults\n            to None, in this case is equivalent to all ones.\n\n    Returns:\n        grad_inputs(Tensor|Tensors): The gradients for inputs.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import numpy as np\n            >>> import paddle\n\n            >>> paddle.enable_static()\n            >>> paddle.incubate.autograd.enable_prim()\n\n            >>> startup_program = paddle.static.Program()\n            >>> main_program = paddle.static.Program()\n            >>> with paddle.static.program_guard(main_program, startup_program):\n            ...     x = paddle.static.data('x', shape=[1], dtype='float32')\n            ...     x.stop_gradients = False\n            ...     y = x * x\n            ...     x_grad = paddle.incubate.autograd.grad(y, x)\n            ...     paddle.incubate.autograd.prim2orig()\n            ...\n            >>> exe = paddle.static.Executor()\n            >>> exe.run(startup_program)\n            >>> x_grad = exe.run(main_program, feed={'x': np.array([2.]).astype('float32')}, fetch_list=[x_grad])\n            >>> print(x_grad)\n            [array([4.], dtype=float32)]\n\n            >>> paddle.incubate.autograd.disable_prim()\n            >>> paddle.disable_static()\n    "
    if not utils.prim_enabled():
        grad_inputs = backward.gradients(outputs, inputs, grad_outputs)
        if isinstance(inputs, framework.Variable) and isinstance(grad_inputs, typing.Sequence) and (len(grad_inputs) > 0):
            return grad_inputs[0]
        else:
            return grad_inputs
    if not isinstance(outputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected outputs is Tensor|Sequence[Tesnor], but got {type(outputs)}.')
    if not isinstance(inputs, (framework.Variable, typing.Sequence)):
        raise TypeError(f'Expected inputs is Tensor|Sequence[Tesnor], but got {type(inputs)}.')
    (ys, xs, ys_bar) = (utils.as_tensors(outputs), utils.as_tensors(inputs), utils.as_tensors(grad_outputs))
    block = framework.default_main_program().current_block()
    if any((x is not None and x.block != block for x in xs + ys)):
        raise RuntimeError('Variable in inputs and outputs should be None or in current block of main program')
    primx.orig2prim(block)
    ad = primx.Transform(block)
    (xs_dot, ys_dot) = ad.linearize(xs, ys)
    if any((var is None for var in ys_dot)):
        raise RuntimeError('Grads cannot be computed. The given outputs does not depend on inputs')
    (ys_bar, xs_bar) = ad.transpose(ys_dot, xs_dot, ys_bar)
    op_indexes = []
    for var in xs_dot:
        if var is not None:
            op_index = block.ops.index(var.op)
            if op_index < 0:
                raise ValueError(f'op_index should be greater than or equal to 0, but op_index={op_index}.')
            op_indexes.append(op_index)
    ad.erase_ops(sorted(op_indexes))
    ad.erase_dots(xs_dot)
    return xs_bar[0] if isinstance(inputs, framework.Variable) else xs_bar

@framework.static_only
def to_prim(blocks, blacklist=frozenset(), whitelist=frozenset(), start_idx=-1, backward_length=-1):
    if False:
        while True:
            i = 10
    'Search nonbasic ops which have be registered composite rules and replace them with primitive ops.\n    The operators in blacklist will be excluded from program when lowering into primitives, and only the\n    operators in whitelist will be lowering. The priority of blacklist is higher than whitelist, it means\n    an operator both in blacklist and whitelist will not be lowering.\n\n    The finally set that will be lowering is:\n        (blocks.ops & ops have decomposite rule & whitelist) - blacklist\n\n    Args:\n        blacklist(frozenset): The Operators that will be exclude when lowering into primitives.\n        whitelist(frozenset): Only the operators in whitelist will be lowering into primitives.\n        start_idx(int): If start_idx exceeds -1, ops[start_idx:] will be processed. Default: -1.\n        backward_length(int): If backward_length exceeds -1, ops[:-backward_length] will be processed. Default: -1.\n    '
    if not core._is_fwd_prim_enabled():
        return
    if isinstance(blocks, paddle.base.framework.Block):
        logging.info('Atomize composite op to primitive ops begin.')
        main_program = blocks.program
    elif isinstance(blocks, typing.Sequence):
        for item in blocks:
            if not isinstance(item, paddle.base.framework.Block):
                raise TypeError(f'Expect block or sequence of blocks, but sequence contains {type(item)}.')
        main_program = blocks[0].program
    else:
        raise TypeError(f'Expect block or sequence of blocks, but got {type(blocks)}.')
    if not isinstance(blacklist, (set, frozenset)):
        raise TypeError(f'Expected type of blacklisst is set|frozenset, but got {type(blacklist)}.')
    if not isinstance(whitelist, (set, frozenset)):
        raise TypeError(f'Expected type of whiltelist is set|frozenset, but got {type(whitelist)}.')
    blacklist = prim_config['forward_blacklist'] | blacklist
    with framework.program_guard(main_program):
        logging.info('Lowering composite forward ops begin...')
        if len(blacklist) > 0 and len(whitelist) > 0:
            filter_ = lambda x: x.type in whitelist and x.type not in blacklist
        elif len(blacklist) > 0 and len(whitelist) == 0:
            filter_ = lambda x: x.type not in blacklist
        elif len(blacklist) == 0 and len(whitelist) > 0:
            filter_ = lambda x: x.type in whitelist
        else:
            filter_ = lambda x: True
        primx._lower_composite(blocks, filter_, start_idx=start_idx, backward_length=backward_length)
        replace_ops = prim_config['composite_ops_record']
        logging.info(f'Lowering composite forward ops finish: {replace_ops}')