import paddle
from paddle.base import core
from paddle.base.backward import _append_grad_suffix_
from paddle.base.framework import Variable
from paddle.common_ops_import import LayerHelper, check_type, in_dygraph_mode
from paddle.utils import flatten, map_structure
from .control_flow import BlockGuard, copy_var_to_parent_block

class StaticPyLayerBlockGuard(BlockGuard):

    def __init__(self, block_manager):
        if False:
            return 10
        check_type(block_manager, 'block', StaticPyLayerBlock, 'StaticPyLayerBlockGuard')
        super().__init__(block_manager.helper.main_program)
        self.block_manager = block_manager

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        super().__enter__()
        return self.block_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.block_manager.complete()
        return super().__exit__(exc_type, exc_val, exc_tb)

class StaticPyLayerBlock:

    def __init__(self, inputs, name=None, pylayer_context=None):
        if False:
            return 10
        self.fwd_inputs = [each_input for each_input in inputs if isinstance(each_input, Variable)]
        self.fwd_outputs = []
        self.context = pylayer_context
        self.helper = LayerHelper('static_pylayer_block', name=name)
        self.fwd_op_id = None
        self._forward_block_id = None
        self._backward_block_id = None
        self.var_old_to_new = {}

    def block(self, is_backward_block=False):
        if False:
            for i in range(10):
                print('nop')
        self.is_backward_block = is_backward_block
        return StaticPyLayerBlockGuard(self)

    @property
    def forward_block_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self._forward_block_id

    @property
    def backward_block_index(self):
        if False:
            while True:
                i = 10
        return self._backward_block_id

    @property
    def fwd_op_index(self):
        if False:
            i = 10
            return i + 15
        return self.fwd_op_id

    def complete_forward_block(self):
        if False:
            print('Hello World!')
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)
        self._forward_block_id = inside_block.idx
        step_scope = parent_block.create_var(type=core.VarDesc.VarType.STEP_SCOPES)
        pylayer_op = parent_block.append_op(type='pylayer', inputs={'Input': self.fwd_inputs}, outputs={'Out': self.fwd_outputs, 'Scope': [step_scope]}, attrs={'blocks': [inside_block]})
        self.fwd_op_id = pylayer_op.idx
        self.helper.main_program._sync_with_cpp()

    def complete_backward_block(self):
        if False:
            for i in range(10):
                print('nop')
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)
        self._backward_block_id = inside_block.idx
        for op in inside_block.ops:
            op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
            backward = core.op_proto_and_checker_maker.OpRole.Backward
            op.desc._set_attr(op_role_attr_name, backward)
        inside_block._set_forward_block_idx(self.forward_block_index)
        _rename_var_recursively_(inside_block, self.var_old_to_new)
        forward_block_desc = parent_block.program.block(self.forward_block_index).desc
        backward_block_desc = inside_block.desc
        parent_block.ops[self.fwd_op_index].desc.set_blocks_attr('blocks', [forward_block_desc, backward_block_desc])
        if self.context:
            for var in self.context.saved_vars:
                if not inside_block.has_var(var.name):
                    raise ValueError(f'{var.name} was saved in forward block but could not be found in backward block. Maybe {var.name} was renamed somewhere.')
                inside_block._remove_var(var.name)
        self.helper.main_program._sync_with_cpp()

    def complete(self):
        if False:
            return 10
        if not self.is_backward_block:
            return self.complete_forward_block()
        else:
            return self.complete_backward_block()

def _get_ctx_from_func_(func):
    if False:
        for i in range(10):
            print('nop')
    if func is None:
        return None
    fn_bind_args = getattr(func, 'args', None)
    if fn_bind_args is None:
        return None
    from paddle.jit.dy2static.py_layer import StaticPyLayerContext
    fn_ctx = None
    if len(fn_bind_args) > 0 and isinstance(fn_bind_args[0], StaticPyLayerContext):
        fn_ctx = fn_bind_args[0]
    return fn_ctx

def _rename_var_recursively_(cur_block, var_old_to_new):
    if False:
        print('Hello World!')
    "\n    Rename the var both the Variable instances and all ops' input and output arg names\n    in `cur_block` based on dict `var_old_to_new`.\n    Dict `var_old_to_new` should be the following format:\n    {\n        old_name_0 : new_name_0,\n        old_name_1 : new_name_1,\n        ...\n        old_name_n : new_name_n,\n    }\n    "
    for (old_var_name, new_var_name) in var_old_to_new.items():
        if cur_block.has_var(old_var_name):
            cur_block.desc._rename_var(old_var_name.encode(), new_var_name.encode())
        else:
            for op in cur_block.ops:
                op._rename_input(old_var_name, new_var_name)
                op._rename_output(old_var_name, new_var_name)
    block_attr_names = ['blocks', 'sub_block']
    for op in cur_block.ops:
        for attr_name in op.all_attrs():
            if attr_name not in block_attr_names:
                continue
            if op.attr_type(attr_name) == core.AttrType.BLOCK:
                sub_block_id = op._block_attr_id(attr_name)
                sub_block = cur_block.program.block(sub_block_id)
                _rename_var_recursively_(sub_block, var_old_to_new)
            elif op.attr_type(attr_name) == core.AttrType.BLOCKS:
                sub_blocks_ids = op._blocks_attr_ids(attr_name)
                for sub_block_id in sub_blocks_ids:
                    sub_block = cur_block.program.block(sub_block_id)
                    _rename_var_recursively_(sub_block, var_old_to_new)

def copy_var_from_parent_block(parent_block_var, layer_helper):
    if False:
        return 10
    if not isinstance(parent_block_var, Variable):
        return parent_block_var
    prog = layer_helper.main_program
    current_block = prog.current_block()
    if parent_block_var.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY and current_block._find_var_recursive(parent_block_var.name):
        current_block_var = parent_block_var
    else:
        current_block_var = current_block.create_var(dtype=parent_block_var.dtype, shape=parent_block_var.shape, type=parent_block_var.type)
        paddle.assign(parent_block_var, current_block_var)
    return current_block_var

def static_pylayer(forward_fn, inputs, backward_fn=None, name=None):
    if False:
        i = 10
        return i + 15
    '\n    This API returns ``forward_fn(inputs)``, and two sub-block are created based on\n    the logic of ``forward_fn`` and ``backward_fn``, with the operator ``pylayer``\n    holding information about the two blocks.\n\n    ``forward_fn`` and ``backward_fn`` should return a nest structure of Variables.\n    A nest structure of Variables in PaddlePaddle is Variable(s), or tuple of Variables, or\n    list of Variables.\n\n    Note:\n        1. If ``backward_fn`` is not None, user needs to keep the number of `Variable` inputs to ``forward_fn`` the same as the\n        number of `Variable` outputs to ``backward_fn``, and the number of `Variable` outputs to ``forward_fn``\n        the same as the number of `Variable` inputs to ``backward_fn``.\n\n        2. If ``backward_fn`` is None, ``stop_gradient`` attr of all Variable in ``inputs`` is expected to be True.\n        Otherwise it might get unexpected results in backward propagation.\n\n        3. This API can only be used under static graph mode.\n\n    Args:\n        forward_fn (callable): A callable to be performed in forward propagation\n        inputs (list[Variable]): The list of input Variable to the ``forward_fn``\n        backward_fn (callable, optional): A callable to be performed in backward propagation. Default: None, which means no need to do backward propagation.\n        name (str, optional): The default value is ``None`` . Normally users\n            don\'t have to set this parameter. For more information, please\n            refer to :ref:`api_guide_Name` .\n\n    Returns:\n        Variable|list(Variable)|tuple(Variable): returns the output of ``forward_fn(inputs)``\n\n    Examples:\n        .. code-block:: python\n\n                >>> import paddle\n                >>> import numpy as np\n\n                >>> paddle.enable_static()\n\n                >>> def forward_fn(x):\n                ...     return paddle.exp(x)\n\n                >>> def backward_fn(dy):\n                ...     return 2 * paddle.exp(dy)\n\n                >>> main_program = paddle.static.Program()\n                >>> start_program = paddle.static.Program()\n\n                >>> place = paddle.CPUPlace()\n                >>> exe = paddle.static.Executor(place)\n                >>> with paddle.static.program_guard(main_program, start_program):\n                ...     data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")\n                ...     data.stop_gradient = False\n                ...     ret = paddle.static.nn.static_pylayer(forward_fn, [data], backward_fn)\n                ...     data_grad = paddle.static.gradients([ret], data)[0]\n\n                >>> exe.run(start_program)\n                >>> x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)\n                >>> x, x_grad, y = exe.run(\n                ...     main_program,\n                ...     feed={"X": x},\n                ...     fetch_list=[\n                ...         data.name,\n                ...         data_grad.name,\n                ...         ret.name\n                ...     ],\n                ... )\n\n                >>> print(x)\n                [[1. 2. 3. 4. 5.]]\n                >>> print(x_grad)\n                [[5.4365635 5.4365635 5.4365635 5.4365635 5.4365635]]\n                >>> print(y)\n                [[  2.7182817   7.389056   20.085537   54.59815   148.41316  ]]\n    '
    assert in_dygraph_mode() is False, 'please use PyLayer instead of static_pylayer in dygraph mode'
    assert isinstance(inputs, list)
    if backward_fn is None:
        for input_var in inputs:
            if input_var.stop_gradient is False:
                raise ValueError('``stop_gradient`` attr of all inputs to ``forward_fn`` are expected to be True, when ``backward_fn == None``, but {}.stop_gradient got {}'.format(input_var.name, input_var.stop_gradient))
    fwd_fn_ctx = _get_ctx_from_func_(forward_fn)
    bwd_fn_ctx = _get_ctx_from_func_(backward_fn)
    static_pylayer_context = fwd_fn_ctx if fwd_fn_ctx and fwd_fn_ctx == bwd_fn_ctx else None
    check_type(name, 'name', (str, type(None)), 'base.layers.static_pylayer')
    helper = LayerHelper('static_pylayer', **locals())
    copy_to_parent_func = lambda var: copy_var_to_parent_block(var, helper)
    assert forward_fn is not None and callable(forward_fn)
    pylayer_block_manager = StaticPyLayerBlock(inputs, pylayer_context=static_pylayer_context)
    with pylayer_block_manager.block(is_backward_block=False) as mgr:
        origin_output = forward_fn(*inputs)
        if origin_output is not None:
            output = map_structure(copy_to_parent_func, origin_output)
            mgr.fwd_outputs = [x for x in flatten(output) if isinstance(x, Variable)]
        else:
            mgr.fwd_outputs = []
    current_block = helper.main_program.current_block()
    current_block._sync_with_cpp()
    if backward_fn is not None:
        assert callable(backward_fn)
        if origin_output is None:
            output = []
        grad_var_ins = []
        for fwd_var in pylayer_block_manager.fwd_outputs:
            fwd_var_name = fwd_var.name
            bwd_var_name = _append_grad_suffix_(fwd_var_name)
            if not current_block.desc.has_var_recursive(fwd_var_name.encode()):
                raise ValueError("Grad var {} , we can't find its related forward var {}".format(bwd_var_name, fwd_var_name))
            var = current_block.create_var(dtype=fwd_var.dtype, shape=fwd_var.shape, type=fwd_var.type, name=bwd_var_name)
            grad_var_ins.append(var)
        copy_from_parent_func = lambda var: copy_var_from_parent_block(var, helper)
        assert isinstance(grad_var_ins, list)
        with pylayer_block_manager.block(is_backward_block=True) as mgr:
            inside_block_inputs = map_structure(copy_from_parent_func, grad_var_ins)
            grad_origin_output = backward_fn(*inside_block_inputs)
            if grad_origin_output is not None:
                flat_grad_origin = flatten(grad_origin_output)
                forward_input_names = current_block.ops[pylayer_block_manager.fwd_op_index].desc.input_arg_names()
                assert len(forward_input_names) == len(flat_grad_origin), f'needs to keep the number of inputs to ``forward_fn`` the same as the number of outputs to ``backward_fn``,                     but got {len(forward_input_names)} and {len(flat_grad_origin)}'
                for (bwd_output, fwd_input_name) in zip(flat_grad_origin, forward_input_names):
                    if isinstance(bwd_output, Variable):
                        bwd_out_new = _append_grad_suffix_(fwd_input_name)
                        mgr.var_old_to_new[bwd_output.name] = bwd_out_new
        for bwd_var in grad_var_ins:
            current_block._remove_var(bwd_var.name)
    if origin_output is None:
        return None
    return output