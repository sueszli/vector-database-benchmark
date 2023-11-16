import types as pytypes
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import next_label, add_offset_to_labels, replace_vars, remove_dels, rename_labels, find_topo_order, merge_adjacent_blocks, GuardException, require, guard, get_definition, find_callname, find_build_sequence, get_np_ufunc_typ, get_ir_of_code, simplify_CFG, canonicalize_array_math, dead_code_elimination
from numba.core.analysis import compute_cfg_from_blocks, compute_use_defs, compute_live_variables
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
'\nVariable enable_inline_arraycall is only used for testing purpose.\n'
enable_inline_arraycall = True

def callee_ir_validator(func_ir):
    if False:
        i = 10
        return i + 15
    'Checks the IR of a callee is supported for inlining\n    '
    for blk in func_ir.blocks.values():
        for stmt in blk.find_insts(ir.Assign):
            if isinstance(stmt.value, ir.Yield):
                msg = 'The use of yield in a closure is unsupported.'
                raise errors.UnsupportedError(msg, loc=stmt.loc)

def _created_inlined_var_name(function_name, var_name):
    if False:
        return 10
    'Creates a name for an inlined variable based on the function name and the\n    variable name. It does this "safely" to avoid the use of characters that are\n    illegal in python variable names as there are occasions when function\n    generation needs valid python name tokens.'
    inlined_name = f'{function_name}.{var_name}'
    new_name = inlined_name.replace('<', '_').replace('>', '_')
    new_name = new_name.replace('.', '_').replace('$', '_v')
    return new_name

class InlineClosureCallPass(object):
    """InlineClosureCallPass class looks for direct calls to locally defined
    closures, and inlines the body of the closure function to the call site.
    """

    def __init__(self, func_ir, parallel_options, swapped={}, typed=False):
        if False:
            for i in range(10):
                print('nop')
        self.func_ir = func_ir
        self.parallel_options = parallel_options
        self.swapped = swapped
        self._processed_stencils = []
        self.typed = typed

    def run(self):
        if False:
            while True:
                i = 10
        'Run inline closure call pass.\n        '
        pp = postproc.PostProcessor(self.func_ir)
        pp.run(True)
        modified = False
        work_list = list(self.func_ir.blocks.items())
        debug_print = _make_debug_print('InlineClosureCallPass')
        debug_print('START')
        while work_list:
            (_label, block) = work_list.pop()
            for (i, instr) in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        call_name = guard(find_callname, self.func_ir, expr)
                        func_def = guard(get_definition, self.func_ir, expr.func)
                        if guard(self._inline_reduction, work_list, block, i, expr, call_name):
                            modified = True
                            break
                        if guard(self._inline_closure, work_list, block, i, func_def):
                            modified = True
                            break
                        if guard(self._inline_stencil, instr, call_name, func_def):
                            modified = True
        if enable_inline_arraycall:
            if modified:
                merge_adjacent_blocks(self.func_ir.blocks)
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            debug_print('start inline arraycall')
            _debug_dump(cfg)
            loops = cfg.loops()
            sized_loops = [(k, len(loops[k].body)) for k in loops.keys()]
            visited = []
            for (k, s) in sorted(sized_loops, key=lambda tup: tup[1], reverse=True):
                visited.append(k)
                if guard(_inline_arraycall, self.func_ir, cfg, visited, loops[k], self.swapped, self.parallel_options.comprehension, self.typed):
                    modified = True
            if modified:
                _fix_nested_array(self.func_ir)
        if modified:
            cfg = compute_cfg_from_blocks(self.func_ir.blocks)
            for dead in cfg.dead_nodes():
                del self.func_ir.blocks[dead]
            dead_code_elimination(self.func_ir)
            self.func_ir.blocks = rename_labels(self.func_ir.blocks)
        remove_dels(self.func_ir.blocks)
        debug_print('END')

    def _inline_reduction(self, work_list, block, i, expr, call_name):
        if False:
            print('Hello World!')
        require(not self.parallel_options.reduction)
        require(call_name == ('reduce', 'builtins') or call_name == ('reduce', '_functools'))
        if len(expr.args) not in (2, 3):
            raise TypeError('invalid reduce call, two arguments are required (optional initial value can also be specified)')
        check_reduce_func(self.func_ir, expr.args[0])

        def reduce_func(f, A, v=None):
            if False:
                while True:
                    i = 10
            it = iter(A)
            if v is not None:
                s = v
            else:
                s = next(it)
            for a in it:
                s = f(s, a)
            return s
        inline_closure_call(self.func_ir, self.func_ir.func_id.func.__globals__, block, i, reduce_func, work_list=work_list, callee_validator=callee_ir_validator)
        return True

    def _inline_stencil(self, instr, call_name, func_def):
        if False:
            for i in range(10):
                print('nop')
        from numba.stencils.stencil import StencilFunc
        lhs = instr.target
        expr = instr.value
        if isinstance(func_def, ir.Global) and func_def.name == 'stencil' and isinstance(func_def.value, StencilFunc):
            if expr.kws:
                expr.kws += func_def.value.kws
            else:
                expr.kws = func_def.value.kws
            return True
        require(call_name == ('stencil', 'numba.stencils.stencil') or call_name == ('stencil', 'numba'))
        require(expr not in self._processed_stencils)
        self._processed_stencils.append(expr)
        if not len(expr.args) == 1:
            raise ValueError('As a minimum Stencil requires a kernel as an argument')
        stencil_def = guard(get_definition, self.func_ir, expr.args[0])
        require(isinstance(stencil_def, ir.Expr) and stencil_def.op == 'make_function')
        kernel_ir = get_ir_of_code(self.func_ir.func_id.func.__globals__, stencil_def.code)
        options = dict(expr.kws)
        if 'neighborhood' in options:
            fixed = guard(self._fix_stencil_neighborhood, options)
            if not fixed:
                raise ValueError('stencil neighborhood option should be a tuple with constant structure such as ((-w, w),)')
        if 'index_offsets' in options:
            fixed = guard(self._fix_stencil_index_offsets, options)
            if not fixed:
                raise ValueError('stencil index_offsets option should be a tuple with constant structure such as (offset, )')
        sf = StencilFunc(kernel_ir, 'constant', options)
        sf.kws = expr.kws
        sf_global = ir.Global('stencil', sf, expr.loc)
        self.func_ir._definitions[lhs.name] = [sf_global]
        instr.value = sf_global
        return True

    def _fix_stencil_neighborhood(self, options):
        if False:
            return 10
        '\n        Extract the two-level tuple representing the stencil neighborhood\n        from the program IR to provide a tuple to StencilFunc.\n        '
        dims_build_tuple = get_definition(self.func_ir, options['neighborhood'])
        require(hasattr(dims_build_tuple, 'items'))
        res = []
        for window_var in dims_build_tuple.items:
            win_build_tuple = get_definition(self.func_ir, window_var)
            require(hasattr(win_build_tuple, 'items'))
            res.append(tuple(win_build_tuple.items))
        options['neighborhood'] = tuple(res)
        return True

    def _fix_stencil_index_offsets(self, options):
        if False:
            return 10
        '\n        Extract the tuple representing the stencil index offsets\n        from the program IR to provide to StencilFunc.\n        '
        offset_tuple = get_definition(self.func_ir, options['index_offsets'])
        require(hasattr(offset_tuple, 'items'))
        options['index_offsets'] = tuple(offset_tuple.items)
        return True

    def _inline_closure(self, work_list, block, i, func_def):
        if False:
            return 10
        require(isinstance(func_def, ir.Expr) and func_def.op == 'make_function')
        inline_closure_call(self.func_ir, self.func_ir.func_id.func.__globals__, block, i, func_def, work_list=work_list, callee_validator=callee_ir_validator)
        return True

def check_reduce_func(func_ir, func_var):
    if False:
        print('Hello World!')
    "Checks the function at func_var in func_ir to make sure it's amenable\n    for inlining. Returns the function itself"
    reduce_func = guard(get_definition, func_ir, func_var)
    if reduce_func is None:
        raise ValueError('Reduce function cannot be found for njit                             analysis')
    if isinstance(reduce_func, (ir.FreeVar, ir.Global)):
        if not isinstance(reduce_func.value, numba.core.registry.CPUDispatcher):
            raise ValueError('Invalid reduction function')
        reduce_func = reduce_func.value.py_func
    elif not (hasattr(reduce_func, 'code') or hasattr(reduce_func, '__code__')):
        raise ValueError('Invalid reduction function')
    f_code = reduce_func.code if hasattr(reduce_func, 'code') else reduce_func.__code__
    if not f_code.co_argcount == 2:
        raise TypeError('Reduction function should take 2 arguments')
    return reduce_func

class InlineWorker(object):
    """ A worker class for inlining, this is a more advanced version of
    `inline_closure_call` in that it permits inlining from function type, Numba
    IR and code object. It also, runs the entire untyped compiler pipeline on
    the inlinee to ensure that it is transformed as though it were compiled
    directly.
    """

    def __init__(self, typingctx=None, targetctx=None, locals=None, pipeline=None, flags=None, validator=callee_ir_validator, typemap=None, calltypes=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Instantiate a new InlineWorker, all arguments are optional though some\n        must be supplied together for certain use cases. The methods will refuse\n        to run if the object isn't configured in the manner needed. Args are the\n        same as those in a numba.core.Compiler.state, except the validator which\n        is a function taking Numba IR and validating it for use when inlining\n        (this is optional and really to just provide better error messages about\n        things which the inliner cannot handle like yield in closure).\n        "

        def check(arg, name):
            if False:
                for i in range(10):
                    print('nop')
            if arg is None:
                raise TypeError('{} must not be None'.format(name))
        from numba.core.compiler import DefaultPassBuilder
        compiler_args = (targetctx, locals, pipeline, flags)
        compiler_group = [x is not None for x in compiler_args]
        if any(compiler_group) and (not all(compiler_group)):
            check(targetctx, 'targetctx')
            check(locals, 'locals')
            check(pipeline, 'pipeline')
            check(flags, 'flags')
        elif all(compiler_group):
            check(typingctx, 'typingctx')
        self._compiler_pipeline = DefaultPassBuilder.define_untyped_pipeline
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.locals = locals
        self.pipeline = pipeline
        self.flags = flags
        self.validator = validator
        self.debug_print = _make_debug_print('InlineWorker')
        pair = (typemap, calltypes)
        pair_is_none = [x is None for x in pair]
        if any(pair_is_none) and (not all(pair_is_none)):
            msg = 'typemap and calltypes must both be either None or have a value, got: %s, %s'
            raise TypeError(msg % pair)
        self._permit_update_type_and_call_maps = not all(pair_is_none)
        self.typemap = typemap
        self.calltypes = calltypes

    def inline_ir(self, caller_ir, block, i, callee_ir, callee_freevars, arg_typs=None):
        if False:
            i = 10
            return i + 15
        ' Inlines the callee_ir in the caller_ir at statement index i of block\n        `block`, callee_freevars are the free variables for the callee_ir. If\n        the callee_ir is derived from a function `func` then this is\n        `func.__code__.co_freevars`. If `arg_typs` is given and the InlineWorker\n        instance was initialized with a typemap and calltypes then they will be\n        appropriately updated based on the arg_typs.\n        '

        def copy_ir(the_ir):
            if False:
                print('Hello World!')
            kernel_copy = the_ir.copy()
            kernel_copy.blocks = {}
            for (block_label, block) in the_ir.blocks.items():
                new_block = copy.deepcopy(the_ir.blocks[block_label])
                kernel_copy.blocks[block_label] = new_block
            return kernel_copy
        callee_ir = copy_ir(callee_ir)
        if self.validator is not None:
            self.validator(callee_ir)
        callee_ir_original = copy_ir(callee_ir)
        scope = block.scope
        instr = block.body[i]
        call_expr = instr.value
        callee_blocks = callee_ir.blocks
        max_label = max(ir_utils._the_max_label.next(), max(caller_ir.blocks.keys()))
        callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
        callee_blocks = simplify_CFG(callee_blocks)
        callee_ir.blocks = callee_blocks
        min_label = min(callee_blocks.keys())
        max_label = max(callee_blocks.keys())
        ir_utils._the_max_label.update(max_label)
        self.debug_print('After relabel')
        _debug_dump(callee_ir)
        callee_scopes = _get_all_scopes(callee_blocks)
        self.debug_print('callee_scopes = ', callee_scopes)
        assert len(callee_scopes) == 1
        callee_scope = callee_scopes[0]
        var_dict = {}
        for var in tuple(callee_scope.localvars._con.values()):
            if not var.name in callee_freevars:
                inlined_name = _created_inlined_var_name(callee_ir.func_id.unique_name, var.name)
                new_var = scope.redefine(inlined_name, loc=var.loc)
                callee_scope.redefine(inlined_name, loc=var.loc)
                var_dict[var.name] = new_var
        self.debug_print('var_dict = ', var_dict)
        replace_vars(callee_blocks, var_dict)
        self.debug_print('After local var rename')
        _debug_dump(callee_ir)
        callee_func = callee_ir.func_id.func
        args = _get_callee_args(call_expr, callee_func, block.body[i].loc, caller_ir)
        if self._permit_update_type_and_call_maps:
            if arg_typs is None:
                raise TypeError('arg_typs should have a value not None')
            self.update_type_and_call_maps(callee_ir, arg_typs)
            callee_blocks = callee_ir.blocks
        self.debug_print('After arguments rename: ')
        _debug_dump(callee_ir)
        _replace_args_with(callee_blocks, args)
        new_blocks = []
        new_block = ir.Block(scope, block.loc)
        new_block.body = block.body[i + 1:]
        new_label = next_label()
        caller_ir.blocks[new_label] = new_block
        new_blocks.append((new_label, new_block))
        block.body = block.body[:i]
        block.body.append(ir.Jump(min_label, instr.loc))
        topo_order = find_topo_order(callee_blocks)
        _replace_returns(callee_blocks, instr.target, new_label)
        if instr.target.name in caller_ir._definitions and call_expr in caller_ir._definitions[instr.target.name]:
            caller_ir._definitions[instr.target.name].remove(call_expr)
        for label in topo_order:
            block = callee_blocks[label]
            block.scope = scope
            _add_definitions(caller_ir, block)
            caller_ir.blocks[label] = block
            new_blocks.append((label, block))
        self.debug_print('After merge in')
        _debug_dump(caller_ir)
        return (callee_ir_original, callee_blocks, var_dict, new_blocks)

    def inline_function(self, caller_ir, block, i, function, arg_typs=None):
        if False:
            return 10
        ' Inlines the function in the caller_ir at statement index i of block\n        `block`. If `arg_typs` is given and the InlineWorker instance was\n        initialized with a typemap and calltypes then they will be appropriately\n        updated based on the arg_typs.\n        '
        callee_ir = self.run_untyped_passes(function)
        freevars = function.__code__.co_freevars
        return self.inline_ir(caller_ir, block, i, callee_ir, freevars, arg_typs=arg_typs)

    def run_untyped_passes(self, func, enable_ssa=False):
        if False:
            i = 10
            return i + 15
        "\n        Run the compiler frontend's untyped passes over the given Python\n        function, and return the function's canonical Numba IR.\n\n        Disable SSA transformation by default, since the call site won't be in\n        SSA form and self.inline_ir depends on this being the case.\n        "
        from numba.core.compiler import StateDict, _CompileStatus
        from numba.core.untyped_passes import ExtractByteCode
        from numba.core import bytecode
        from numba.parfors.parfor import ParforDiagnostics
        state = StateDict()
        state.func_ir = None
        state.typingctx = self.typingctx
        state.targetctx = self.targetctx
        state.locals = self.locals
        state.pipeline = self.pipeline
        state.flags = self.flags
        state.flags.enable_ssa = enable_ssa
        state.func_id = bytecode.FunctionIdentity.from_function(func)
        state.typemap = None
        state.calltypes = None
        state.type_annotation = None
        state.status = _CompileStatus(False)
        state.return_type = None
        state.parfor_diagnostics = ParforDiagnostics()
        state.metadata = {}
        ExtractByteCode().run_pass(state)
        state.args = len(state.bc.func_id.pysig.parameters) * (types.pyobject,)
        pm = self._compiler_pipeline(state)
        pm.finalize()
        pm.run(state)
        return state.func_ir

    def update_type_and_call_maps(self, callee_ir, arg_typs):
        if False:
            for i in range(10):
                print('nop')
        ' Updates the type and call maps based on calling callee_ir with\n        arguments from arg_typs'
        from numba.core.ssa import reconstruct_ssa
        from numba.core.typed_passes import PreLowerStripPhis
        if not self._permit_update_type_and_call_maps:
            msg = 'InlineWorker instance not configured correctly, typemap or calltypes missing in initialization.'
            raise ValueError(msg)
        from numba.core import typed_passes
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
        callee_ir = reconstruct_ssa(callee_ir)
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        [f_typemap, _f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(self.typingctx, self.targetctx, callee_ir, arg_typs, None)
        callee_ir = PreLowerStripPhis()._strip_phi_nodes(callee_ir)
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        canonicalize_array_math(callee_ir, f_typemap, f_calltypes, self.typingctx)
        arg_names = [vname for vname in f_typemap if vname.startswith('arg.')]
        for a in arg_names:
            f_typemap.pop(a)
        self.typemap.update(f_typemap)
        self.calltypes.update(f_calltypes)

def inline_closure_call(func_ir, glbls, block, i, callee, typingctx=None, targetctx=None, arg_typs=None, typemap=None, calltypes=None, work_list=None, callee_validator=None, replace_freevars=True):
    if False:
        return 10
    "Inline the body of `callee` at its callsite (`i`-th instruction of\n    `block`)\n\n    `func_ir` is the func_ir object of the caller function and `glbls` is its\n    global variable environment (func_ir.func_id.func.__globals__).\n    `block` is the IR block of the callsite and `i` is the index of the\n    callsite's node. `callee` is either the called function or a\n    make_function node. `typingctx`, `typemap` and `calltypes` are typing\n    data structures of the caller, available if we are in a typed pass.\n    `arg_typs` includes the types of the arguments at the callsite.\n    `callee_validator` is an optional callable which can be used to validate the\n    IR of the callee to ensure that it contains IR supported for inlining, it\n    takes one argument, the func_ir of the callee\n\n    Returns IR blocks of the callee and the variable renaming dictionary used\n    for them to facilitate further processing of new blocks.\n    "
    scope = block.scope
    instr = block.body[i]
    call_expr = instr.value
    debug_print = _make_debug_print('inline_closure_call')
    debug_print('Found closure call: ', instr, ' with callee = ', callee)
    callee_code = callee.code if hasattr(callee, 'code') else callee.__code__
    callee_closure = callee.closure if hasattr(callee, 'closure') else callee.__closure__
    if isinstance(callee, pytypes.FunctionType):
        from numba.core import compiler
        callee_ir = compiler.run_frontend(callee, inline_closures=True)
    else:
        callee_ir = get_ir_of_code(glbls, callee_code)
    if callee_validator is not None:
        callee_validator(callee_ir)
    callee_blocks = callee_ir.blocks
    max_label = max(ir_utils._the_max_label.next(), max(func_ir.blocks.keys()))
    callee_blocks = add_offset_to_labels(callee_blocks, max_label + 1)
    callee_blocks = simplify_CFG(callee_blocks)
    callee_ir.blocks = callee_blocks
    min_label = min(callee_blocks.keys())
    max_label = max(callee_blocks.keys())
    ir_utils._the_max_label.update(max_label)
    debug_print('After relabel')
    _debug_dump(callee_ir)
    callee_scopes = _get_all_scopes(callee_blocks)
    debug_print('callee_scopes = ', callee_scopes)
    assert len(callee_scopes) == 1
    callee_scope = callee_scopes[0]
    var_dict = {}
    for var in callee_scope.localvars._con.values():
        if not var.name in callee_code.co_freevars:
            inlined_name = _created_inlined_var_name(callee_ir.func_id.unique_name, var.name)
            new_var = scope.redefine(inlined_name, loc=var.loc)
            var_dict[var.name] = new_var
    debug_print('var_dict = ', var_dict)
    replace_vars(callee_blocks, var_dict)
    debug_print('After local var rename')
    _debug_dump(callee_ir)
    args = _get_callee_args(call_expr, callee, block.body[i].loc, func_ir)
    debug_print('After arguments rename: ')
    _debug_dump(callee_ir)
    if callee_closure and replace_freevars:
        closure = func_ir.get_definition(callee_closure)
        debug_print("callee's closure = ", closure)
        if isinstance(closure, tuple):
            cellget = ctypes.pythonapi.PyCell_Get
            cellget.restype = ctypes.py_object
            cellget.argtypes = (ctypes.py_object,)
            items = tuple((cellget(x) for x in closure))
        else:
            assert isinstance(closure, ir.Expr) and closure.op == 'build_tuple'
            items = closure.items
        assert len(callee_code.co_freevars) == len(items)
        _replace_freevars(callee_blocks, items)
        debug_print('After closure rename')
        _debug_dump(callee_ir)
    if typingctx:
        from numba.core import typed_passes
        callee_ir._definitions = ir_utils.build_definitions(callee_ir.blocks)
        numba.core.analysis.dead_branch_prune(callee_ir, arg_typs)
        try:
            [f_typemap, f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(typingctx, targetctx, callee_ir, arg_typs, None)
        except Exception:
            [f_typemap, f_return_type, f_calltypes, _] = typed_passes.type_inference_stage(typingctx, targetctx, callee_ir, arg_typs, None)
        canonicalize_array_math(callee_ir, f_typemap, f_calltypes, typingctx)
        arg_names = [vname for vname in f_typemap if vname.startswith('arg.')]
        for a in arg_names:
            f_typemap.pop(a)
        typemap.update(f_typemap)
        calltypes.update(f_calltypes)
    _replace_args_with(callee_blocks, args)
    new_blocks = []
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1:]
    new_label = next_label()
    func_ir.blocks[new_label] = new_block
    new_blocks.append((new_label, new_block))
    block.body = block.body[:i]
    block.body.append(ir.Jump(min_label, instr.loc))
    topo_order = find_topo_order(callee_blocks)
    _replace_returns(callee_blocks, instr.target, new_label)
    if instr.target.name in func_ir._definitions and call_expr in func_ir._definitions[instr.target.name]:
        func_ir._definitions[instr.target.name].remove(call_expr)
    for label in topo_order:
        block = callee_blocks[label]
        block.scope = scope
        _add_definitions(func_ir, block)
        func_ir.blocks[label] = block
        new_blocks.append((label, block))
    debug_print('After merge in')
    _debug_dump(func_ir)
    if work_list is not None:
        for block in new_blocks:
            work_list.append(block)
    return (callee_blocks, var_dict)

def _get_callee_args(call_expr, callee, loc, func_ir):
    if False:
        for i in range(10):
            print('nop')
    "Get arguments for calling 'callee', including the default arguments.\n    keyword arguments are currently only handled when 'callee' is a function.\n    "
    if call_expr.op == 'call':
        args = list(call_expr.args)
        if call_expr.vararg:
            msg = 'Calling a closure with *args is unsupported.'
            raise errors.UnsupportedError(msg, call_expr.loc)
    elif call_expr.op == 'getattr':
        args = [call_expr.value]
    elif ir_utils.is_operator_or_getitem(call_expr):
        args = call_expr.list_vars()
    else:
        raise TypeError('Unsupported ir.Expr.{}'.format(call_expr.op))
    debug_print = _make_debug_print('inline_closure_call default handling')
    if isinstance(callee, pytypes.FunctionType):
        pysig = numba.core.utils.pysignature(callee)
        normal_handler = lambda index, param, default: default
        default_handler = lambda index, param, default: ir.Const(default, loc)

        def stararg_handler(index, param, default):
            if False:
                return 10
            raise NotImplementedError('Stararg not supported in inliner for arg {} {}'.format(index, param))
        if call_expr.op == 'call':
            kws = dict(call_expr.kws)
        else:
            kws = {}
        return numba.core.typing.fold_arguments(pysig, args, kws, normal_handler, default_handler, stararg_handler)
    else:
        callee_defaults = callee.defaults if hasattr(callee, 'defaults') else callee.__defaults__
        if callee_defaults:
            debug_print('defaults = ', callee_defaults)
            if isinstance(callee_defaults, tuple):
                defaults_list = []
                for x in callee_defaults:
                    if isinstance(x, ir.Var):
                        defaults_list.append(x)
                    else:
                        defaults_list.append(ir.Const(value=x, loc=loc))
                args = args + defaults_list
            elif isinstance(callee_defaults, ir.Var) or isinstance(callee_defaults, str):
                default_tuple = func_ir.get_definition(callee_defaults)
                assert isinstance(default_tuple, ir.Expr)
                assert default_tuple.op == 'build_tuple'
                const_vals = [func_ir.get_definition(x) for x in default_tuple.items]
                args = args + const_vals
            else:
                raise NotImplementedError('Unsupported defaults to make_function: {}'.format(callee_defaults))
        return args

def _make_debug_print(prefix):
    if False:
        return 10

    def debug_print(*args):
        if False:
            print('Hello World!')
        if config.DEBUG_INLINE_CLOSURE:
            print(prefix + ': ' + ''.join((str(x) for x in args)))
    return debug_print

def _debug_dump(func_ir):
    if False:
        while True:
            i = 10
    if config.DEBUG_INLINE_CLOSURE:
        func_ir.dump()

def _get_all_scopes(blocks):
    if False:
        for i in range(10):
            print('nop')
    'Get all block-local scopes from an IR.\n    '
    all_scopes = []
    for (label, block) in blocks.items():
        if not block.scope in all_scopes:
            all_scopes.append(block.scope)
    return all_scopes

def _replace_args_with(blocks, args):
    if False:
        while True:
            i = 10
    '\n    Replace ir.Arg(...) with real arguments from call site\n    '
    for (label, block) in blocks.items():
        assigns = block.find_insts(ir.Assign)
        for stmt in assigns:
            if isinstance(stmt.value, ir.Arg):
                idx = stmt.value.index
                assert idx < len(args)
                stmt.value = args[idx]

def _replace_freevars(blocks, args):
    if False:
        i = 10
        return i + 15
    '\n    Replace ir.FreeVar(...) with real variables from parent function\n    '
    for (label, block) in blocks.items():
        assigns = block.find_insts(ir.Assign)
        for stmt in assigns:
            if isinstance(stmt.value, ir.FreeVar):
                idx = stmt.value.index
                assert idx < len(args)
                if isinstance(args[idx], ir.Var):
                    stmt.value = args[idx]
                else:
                    stmt.value = ir.Const(args[idx], stmt.loc)

def _replace_returns(blocks, target, return_label):
    if False:
        while True:
            i = 10
    '\n    Return return statement by assigning directly to target, and a jump.\n    '
    for (label, block) in blocks.items():
        casts = []
        for i in range(len(block.body)):
            stmt = block.body[i]
            if isinstance(stmt, ir.Return):
                assert i + 1 == len(block.body)
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))
                for cast in casts:
                    if cast.target.name == stmt.value.name:
                        cast.value = cast.value.value
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'cast'):
                casts.append(stmt)

def _add_definitions(func_ir, block):
    if False:
        print('Hello World!')
    '\n    Add variable definitions found in a block to parent func_ir.\n    '
    definitions = func_ir._definitions
    assigns = block.find_insts(ir.Assign)
    for stmt in assigns:
        definitions[stmt.target.name].append(stmt.value)

def _find_arraycall(func_ir, block):
    if False:
        while True:
            i = 10
    'Look for statement like "x = numpy.array(y)" or "x[..] = y"\n    immediately after the closure call that creates list y (the i-th\n    statement in block).  Return the statement index if found, or\n    raise GuardException.\n    '
    array_var = None
    list_var_dead_after_array_call = False
    list_var = None
    i = 0
    while i < len(block.body):
        instr = block.body[i]
        if isinstance(instr, ir.Del):
            if list_var and array_var and (instr.value == list_var.name):
                list_var_dead_after_array_call = True
                break
            pass
        elif isinstance(instr, ir.Assign):
            lhs = instr.target
            expr = instr.value
            if guard(find_callname, func_ir, expr) == ('array', 'numpy') and isinstance(expr.args[0], ir.Var):
                list_var = expr.args[0]
                array_var = lhs
                array_stmt_index = i
                array_kws = dict(expr.kws)
        elif isinstance(instr, ir.SetItem) and isinstance(instr.value, ir.Var) and (not list_var):
            list_var = instr.value
            array_var = instr.target
            array_def = get_definition(func_ir, array_var)
            require(guard(_find_unsafe_empty_inferred, func_ir, array_def))
            array_stmt_index = i
            array_kws = {}
        else:
            break
        i = i + 1
    require(array_var and list_var_dead_after_array_call)
    _make_debug_print('find_array_call')(block.body[array_stmt_index])
    return (list_var, array_stmt_index, array_kws)

def _find_iter_range(func_ir, range_iter_var, swapped):
    if False:
        for i in range(10):
            print('nop')
    "Find the iterator's actual range if it is either range(n), or\n    range(m, n), otherwise return raise GuardException.\n    "
    debug_print = _make_debug_print('find_iter_range')
    range_iter_def = get_definition(func_ir, range_iter_var)
    debug_print('range_iter_var = ', range_iter_var, ' def = ', range_iter_def)
    require(isinstance(range_iter_def, ir.Expr) and range_iter_def.op == 'getiter')
    range_var = range_iter_def.value
    range_def = get_definition(func_ir, range_var)
    debug_print('range_var = ', range_var, ' range_def = ', range_def)
    require(isinstance(range_def, ir.Expr) and range_def.op == 'call')
    func_var = range_def.func
    func_def = get_definition(func_ir, func_var)
    debug_print('func_var = ', func_var, ' func_def = ', func_def)
    require(isinstance(func_def, ir.Global) and (func_def.value == range or func_def.value == numba.misc.special.prange))
    nargs = len(range_def.args)
    swapping = [('"array comprehension"', 'closure of'), range_def.func.loc]
    if nargs == 1:
        swapped[range_def.func.name] = swapping
        stop = get_definition(func_ir, range_def.args[0], lhs_only=True)
        return (0, range_def.args[0], func_def)
    elif nargs == 2:
        swapped[range_def.func.name] = swapping
        start = get_definition(func_ir, range_def.args[0], lhs_only=True)
        stop = get_definition(func_ir, range_def.args[1], lhs_only=True)
        return (start, stop, func_def)
    else:
        raise GuardException

def _inline_arraycall(func_ir, cfg, visited, loop, swapped, enable_prange=False, typed=False):
    if False:
        i = 10
        return i + 15
    'Look for array(list) call in the exit block of a given loop, and turn\n    list operations into array operations in the loop if the following\n    conditions are met:\n      1. The exit block contains an array call on the list;\n      2. The list variable is no longer live after array call;\n      3. The list is created in the loop entry block;\n      4. The loop is created from an range iterator whose length is known prior\n         to the loop;\n      5. There is only one list_append operation on the list variable in the\n         loop body;\n      6. The block that contains list_append dominates the loop head, which\n         ensures list length is the same as loop length;\n    If any condition check fails, no modification will be made to the incoming\n    IR.\n    '
    debug_print = _make_debug_print('inline_arraycall')
    require(len(loop.exits) == 1)
    exit_block = next(iter(loop.exits))
    (list_var, array_call_index, array_kws) = _find_arraycall(func_ir, func_ir.blocks[exit_block])
    dtype_def = None
    dtype_mod_def = None
    if 'dtype' in array_kws:
        require(isinstance(array_kws['dtype'], ir.Var))
        dtype_def = get_definition(func_ir, array_kws['dtype'])
        require(isinstance(dtype_def, ir.Expr) and dtype_def.op == 'getattr')
        dtype_mod_def = get_definition(func_ir, dtype_def.value)
    list_var_def = get_definition(func_ir, list_var)
    debug_print('list_var = ', list_var, ' def = ', list_var_def)
    if isinstance(list_var_def, ir.Expr) and list_var_def.op == 'cast':
        list_var_def = get_definition(func_ir, list_var_def.value)
    require(isinstance(list_var_def, ir.Expr) and list_var_def.op == 'build_list')
    require(len(list_var_def.items) == 0)
    list_append_stmts = []
    for label in loop.body:
        in_visited_loops = [l.header in visited for l in cfg.in_loops(label)]
        if not all(in_visited_loops):
            continue
        block = func_ir.blocks[label]
        debug_print('check loop body block ', label)
        for stmt in block.find_insts(ir.Assign):
            expr = stmt.value
            if isinstance(expr, ir.Expr) and expr.op == 'call':
                func_def = get_definition(func_ir, expr.func)
                if isinstance(func_def, ir.Expr) and func_def.op == 'getattr' and (func_def.attr == 'append'):
                    list_def = get_definition(func_ir, func_def.value)
                    debug_print('list_def = ', list_def, list_def is list_var_def)
                    if list_def is list_var_def:
                        list_append_stmts.append((label, block, stmt))
    require(len(list_append_stmts) == 1)
    (append_block_label, append_block, append_stmt) = list_append_stmts[0]
    preds = set((l for (l, b) in cfg.predecessors(loop.header)))
    debug_print('preds = ', preds, loop.entries | set([append_block_label]))
    require(preds == loop.entries | set([append_block_label]))
    iter_vars = []
    iter_first_vars = []
    loop_header = func_ir.blocks[loop.header]
    for stmt in loop_header.find_insts(ir.Assign):
        expr = stmt.value
        if isinstance(expr, ir.Expr):
            if expr.op == 'iternext':
                iter_def = get_definition(func_ir, expr.value)
                debug_print('iter_def = ', iter_def)
                iter_vars.append(expr.value)
            elif expr.op == 'pair_first':
                iter_first_vars.append(stmt.target)
    require(len(iter_vars) == 1 and len(iter_first_vars) == 1)
    iter_var = iter_vars[0]
    iter_first_var = iter_first_vars[0]
    require(len(loop.entries) == 1)
    loop_entry = func_ir.blocks[next(iter(loop.entries))]
    terminator = loop_entry.terminator
    scope = loop_entry.scope
    loc = loop_entry.loc
    stmts = []
    removed = []

    def is_removed(val, removed):
        if False:
            print('Hello World!')
        if isinstance(val, ir.Var):
            for x in removed:
                if x.name == val.name:
                    return True
        return False
    for i in range(len(loop_entry.body) - 1):
        stmt = loop_entry.body[i]
        if isinstance(stmt, ir.Assign) and (stmt.value is list_def or is_removed(stmt.value, removed)):
            removed.append(stmt.target)
        else:
            stmts.append(stmt)
    debug_print('removed variables: ', removed)
    range_def = guard(_find_iter_range, func_ir, iter_var, swapped)
    index_var = scope.redefine('index', loc)
    if range_def and range_def[0] == 0:
        index_var = iter_first_var
    else:
        stmts.append(_new_definition(func_ir, index_var, ir.Const(value=-1, loc=loc), loc))
    size_var = scope.redefine('size', loc)
    if range_def:
        (start, stop, range_func_def) = range_def
        if start == 0:
            size_val = stop
        else:
            size_val = ir.Expr.binop(fn=operator.sub, lhs=stop, rhs=start, loc=loc)
        if enable_prange and isinstance(range_func_def, ir.Global):
            range_func_def.name = 'internal_prange'
            range_func_def.value = internal_prange
    elif typed:
        len_func_var = scope.redefine('len_func', loc)
        from numba.cpython.rangeobj import length_of_iterator
        stmts.append(_new_definition(func_ir, len_func_var, ir.Global('length_of_iterator', length_of_iterator, loc=loc), loc))
        size_val = ir.Expr.call(len_func_var, (iter_var,), (), loc=loc)
    else:
        raise GuardException
    stmts.append(_new_definition(func_ir, size_var, size_val, loc))
    size_tuple_var = scope.redefine('size_tuple', loc)
    stmts.append(_new_definition(func_ir, size_tuple_var, ir.Expr.build_tuple(items=[size_var], loc=loc), loc))
    array_var = scope.redefine('array', loc)
    empty_func = scope.redefine('empty_func', loc)
    if dtype_def and dtype_mod_def:
        dtype_mod_var = scope.redefine('dtype_mod', loc)
        dtype_var = scope.redefine('dtype', loc)
        stmts.append(_new_definition(func_ir, dtype_mod_var, dtype_mod_def, loc))
        stmts.append(_new_definition(func_ir, dtype_var, ir.Expr.getattr(dtype_mod_var, dtype_def.attr, loc), loc))
        stmts.append(_new_definition(func_ir, empty_func, ir.Global('empty', np.empty, loc=loc), loc))
        array_kws = [('dtype', dtype_var)]
    elif typed:
        stmts.append(_new_definition(func_ir, empty_func, ir.Global('unsafe_empty_inferred', unsafe_empty_inferred, loc=loc), loc))
        array_kws = []
    else:
        raise GuardException
    stmts.append(_new_definition(func_ir, array_var, ir.Expr.call(empty_func, (size_tuple_var,), list(array_kws), loc=loc), loc))
    for var in removed:
        stmts.append(_new_definition(func_ir, var, array_var, loc))
    stmts.append(terminator)
    loop_entry.body = stmts
    if range_def:
        if range_def[0] != 0:
            terminator = loop_header.terminator
            assert isinstance(terminator, ir.Branch)
            block_id = terminator.truebr
            blk = func_ir.blocks[block_id]
            loc = blk.loc
            blk.body.insert(0, _new_definition(func_ir, index_var, ir.Expr.binop(fn=operator.sub, lhs=iter_first_var, rhs=range_def[0], loc=loc), loc))
    else:
        loc = loop_header.loc
        terminator = loop_header.terminator
        stmts = loop_header.body[0:-1]
        next_index_var = scope.redefine('next_index', loc)
        one = scope.redefine('one', loc)
        stmts.append(_new_definition(func_ir, one, ir.Const(value=1, loc=loc), loc))
        stmts.append(_new_definition(func_ir, next_index_var, ir.Expr.binop(fn=operator.add, lhs=index_var, rhs=one, loc=loc), loc))
        stmts.append(_new_definition(func_ir, index_var, next_index_var, loc))
        stmts.append(terminator)
        loop_header.body = stmts
    for i in range(len(append_block.body)):
        if append_block.body[i] is append_stmt:
            debug_print('Replace append with SetItem')
            append_block.body[i] = ir.SetItem(target=array_var, index=index_var, value=append_stmt.value.args[0], loc=append_stmt.loc)
    stmt = func_ir.blocks[exit_block].body[array_call_index]
    if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
        stmt.value = array_var
        func_ir._definitions[stmt.target.name] = [stmt.value]
    return True

def _find_unsafe_empty_inferred(func_ir, expr):
    if False:
        while True:
            i = 10
    unsafe_empty_inferred
    require(isinstance(expr, ir.Expr) and expr.op == 'call')
    callee = expr.func
    callee_def = get_definition(func_ir, callee)
    require(isinstance(callee_def, ir.Global))
    _make_debug_print('_find_unsafe_empty_inferred')(callee_def.value)
    return callee_def.value == unsafe_empty_inferred

def _fix_nested_array(func_ir):
    if False:
        i = 10
        return i + 15
    'Look for assignment like: a[..] = b, where both a and b are numpy arrays,\n    and try to eliminate array b by expanding a with an extra dimension.\n    '
    blocks = func_ir.blocks
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    empty_deadmap = dict([(label, set()) for label in blocks.keys()])
    livemap = compute_live_variables(cfg, blocks, usedefs.defmap, empty_deadmap)

    def find_array_def(arr):
        if False:
            while True:
                i = 10
        'Find numpy array definition such as\n            arr = numba.unsafe.ndarray.empty_inferred(...).\n        If it is arr = b[...], find array definition of b recursively.\n        '
        arr_def = get_definition(func_ir, arr)
        _make_debug_print('find_array_def')(arr, arr_def)
        if isinstance(arr_def, ir.Expr):
            if guard(_find_unsafe_empty_inferred, func_ir, arr_def):
                return arr_def
            elif arr_def.op == 'getitem':
                return find_array_def(arr_def.value)
        raise GuardException

    def fix_dependencies(expr, varlist):
        if False:
            while True:
                i = 10
        "Double check if all variables in varlist are defined before\n        expr is used. Try to move constant definition when the check fails.\n        Bails out by raising GuardException if it can't be moved.\n        "
        debug_print = _make_debug_print('fix_dependencies')
        for (label, block) in blocks.items():
            scope = block.scope
            body = block.body
            defined = set()
            for i in range(len(body)):
                inst = body[i]
                if isinstance(inst, ir.Assign):
                    defined.add(inst.target.name)
                    if inst.value is expr:
                        new_varlist = []
                        for var in varlist:
                            if var.name in defined or (var.name in livemap[label] and (not var.name in usedefs.defmap[label])):
                                debug_print(var.name, ' already defined')
                                new_varlist.append(var)
                            else:
                                debug_print(var.name, ' not yet defined')
                                var_def = get_definition(func_ir, var.name)
                                if isinstance(var_def, ir.Const):
                                    loc = var.loc
                                    new_var = scope.redefine('new_var', loc)
                                    new_const = ir.Const(var_def.value, loc)
                                    new_vardef = _new_definition(func_ir, new_var, new_const, loc)
                                    new_body = []
                                    new_body.extend(body[:i])
                                    new_body.append(new_vardef)
                                    new_body.extend(body[i:])
                                    block.body = new_body
                                    new_varlist.append(new_var)
                                else:
                                    raise GuardException
                        return new_varlist
        raise GuardException

    def fix_array_assign(stmt):
        if False:
            i = 10
            return i + 15
        'For assignment like lhs[idx] = rhs, where both lhs and rhs are\n        arrays, do the following:\n        1. find the definition of rhs, which has to be a call to\n           numba.unsafe.ndarray.empty_inferred\n        2. find the source array creation for lhs, insert an extra dimension of\n           size of b.\n        3. replace the definition of\n           rhs = numba.unsafe.ndarray.empty_inferred(...) with rhs = lhs[idx]\n        '
        require(isinstance(stmt, ir.SetItem))
        require(isinstance(stmt.value, ir.Var))
        debug_print = _make_debug_print('fix_array_assign')
        debug_print('found SetItem: ', stmt)
        lhs = stmt.target
        lhs_def = find_array_def(lhs)
        debug_print('found lhs_def: ', lhs_def)
        rhs_def = get_definition(func_ir, stmt.value)
        debug_print('found rhs_def: ', rhs_def)
        require(isinstance(rhs_def, ir.Expr))
        if rhs_def.op == 'cast':
            rhs_def = get_definition(func_ir, rhs_def.value)
            require(isinstance(rhs_def, ir.Expr))
        require(_find_unsafe_empty_inferred(func_ir, rhs_def))
        dim_def = get_definition(func_ir, rhs_def.args[0])
        require(isinstance(dim_def, ir.Expr) and dim_def.op == 'build_tuple')
        debug_print('dim_def = ', dim_def)
        extra_dims = [get_definition(func_ir, x, lhs_only=True) for x in dim_def.items]
        debug_print('extra_dims = ', extra_dims)
        size_tuple_def = get_definition(func_ir, lhs_def.args[0])
        require(isinstance(size_tuple_def, ir.Expr) and size_tuple_def.op == 'build_tuple')
        debug_print('size_tuple_def = ', size_tuple_def)
        extra_dims = fix_dependencies(size_tuple_def, extra_dims)
        size_tuple_def.items += extra_dims
        rhs_def.op = 'getitem'
        rhs_def.fn = operator.getitem
        rhs_def.value = get_definition(func_ir, lhs, lhs_only=True)
        rhs_def.index = stmt.index
        del rhs_def._kws['func']
        del rhs_def._kws['args']
        del rhs_def._kws['vararg']
        del rhs_def._kws['kws']
        return True
    for label in find_topo_order(func_ir.blocks):
        block = func_ir.blocks[label]
        for stmt in block.body:
            if guard(fix_array_assign, stmt):
                block.body.remove(stmt)

def _new_definition(func_ir, var, value, loc):
    if False:
        for i in range(10):
            print('nop')
    func_ir._definitions[var.name] = [value]
    return ir.Assign(value=value, target=var, loc=loc)

@rewrites.register_rewrite('after-inference')
class RewriteArrayOfConsts(rewrites.Rewrite):
    """The RewriteArrayOfConsts class is responsible for finding
    1D array creations from a constant list, and rewriting it into
    direct initialization of array elements without creating the list.
    """

    def __init__(self, state, *args, **kws):
        if False:
            i = 10
            return i + 15
        self.typingctx = state.typingctx
        super(RewriteArrayOfConsts, self).__init__(*args, **kws)

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            i = 10
            return i + 15
        if len(calltypes) == 0:
            return False
        self.crnt_block = block
        self.new_body = guard(_inline_const_arraycall, block, func_ir, self.typingctx, typemap, calltypes)
        return self.new_body is not None

    def apply(self):
        if False:
            return 10
        self.crnt_block.body = self.new_body
        return self.crnt_block

def _inline_const_arraycall(block, func_ir, context, typemap, calltypes):
    if False:
        i = 10
        return i + 15
    'Look for array(list) call where list is a constant list created by\n    build_list, and turn them into direct array creation and initialization, if\n    the following conditions are met:\n      1. The build_list call immediate precedes the array call;\n      2. The list variable is no longer live after array call;\n    If any condition check fails, no modification will be made.\n    '
    debug_print = _make_debug_print('inline_const_arraycall')
    scope = block.scope

    def inline_array(array_var, expr, stmts, list_vars, dels):
        if False:
            print('Hello World!')
        'Check to see if the given "array_var" is created from a list\n        of constants, and try to inline the list definition as array\n        initialization.\n\n        Extra statements produced with be appended to "stmts".\n        '
        callname = guard(find_callname, func_ir, expr)
        require(callname and callname[1] == 'numpy' and (callname[0] == 'array'))
        require(expr.args[0].name in list_vars)
        ret_type = calltypes[expr].return_type
        require(isinstance(ret_type, types.ArrayCompatible) and ret_type.ndim == 1)
        loc = expr.loc
        list_var = expr.args[0]
        array_typ = typemap[array_var.name]
        debug_print('inline array_var = ', array_var, ' list_var = ', list_var)
        dtype = array_typ.dtype
        (seq, _) = find_build_sequence(func_ir, list_var)
        size = len(seq)
        size_var = scope.redefine('size', loc)
        size_tuple_var = scope.redefine('size_tuple', loc)
        size_typ = types.intp
        size_tuple_typ = types.UniTuple(size_typ, 1)
        typemap[size_var.name] = size_typ
        typemap[size_tuple_var.name] = size_tuple_typ
        stmts.append(_new_definition(func_ir, size_var, ir.Const(size, loc=loc), loc))
        stmts.append(_new_definition(func_ir, size_tuple_var, ir.Expr.build_tuple(items=[size_var], loc=loc), loc))
        nptype = types.DType(dtype)
        empty_func = scope.redefine('empty_func', loc)
        fnty = get_np_ufunc_typ(np.empty)
        context.resolve_function_type(fnty, (size_typ,), {'dtype': nptype})
        typemap[empty_func.name] = fnty
        stmts.append(_new_definition(func_ir, empty_func, ir.Global('empty', np.empty, loc=loc), loc))
        g_np_var = scope.redefine('$np_g_var', loc)
        typemap[g_np_var.name] = types.misc.Module(np)
        g_np = ir.Global('np', np, loc)
        stmts.append(_new_definition(func_ir, g_np_var, g_np, loc))
        typ_var = scope.redefine('$np_typ_var', loc)
        typemap[typ_var.name] = nptype
        dtype_str = str(dtype)
        if dtype_str == 'bool':
            dtype_str = 'bool_'
        np_typ_getattr = ir.Expr.getattr(g_np_var, dtype_str, loc)
        stmts.append(_new_definition(func_ir, typ_var, np_typ_getattr, loc))
        empty_call = ir.Expr.call(empty_func, [size_var, typ_var], {}, loc=loc)
        calltypes[empty_call] = typing.signature(array_typ, size_typ, nptype)
        stmts.append(_new_definition(func_ir, array_var, empty_call, loc))
        for i in range(size):
            index_var = scope.redefine('index', loc)
            index_typ = types.intp
            typemap[index_var.name] = index_typ
            stmts.append(_new_definition(func_ir, index_var, ir.Const(i, loc), loc))
            setitem = ir.SetItem(array_var, index_var, seq[i], loc)
            calltypes[setitem] = typing.signature(types.none, array_typ, index_typ, dtype)
            stmts.append(setitem)
        stmts.extend(dels)
        return True

    class State(object):
        """
        This class is used to hold the state in the following loop so as to make
        it easy to reset the state of the variables tracking the various
        statement kinds
        """

        def __init__(self):
            if False:
                return 10
            self.list_vars = []
            self.dead_vars = []
            self.list_items = []
            self.stmts = []
            self.dels = []
            self.modified = False

        def reset(self):
            if False:
                return 10
            '\n            Resets the internal state of the variables used for tracking\n            '
            self.list_vars = []
            self.dead_vars = []
            self.list_items = []
            self.dels = []

        def list_var_used(self, inst):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Returns True if the list being analysed is used between the\n            build_list and the array call.\n            '
            return any([x.name in self.list_vars for x in inst.list_vars()])
    state = State()
    for inst in block.body:
        if isinstance(inst, ir.Assign):
            if isinstance(inst.value, ir.Var):
                if inst.value.name in state.list_vars:
                    state.list_vars.append(inst.target.name)
                    state.stmts.append(inst)
                    continue
            elif isinstance(inst.value, ir.Expr):
                expr = inst.value
                if expr.op == 'build_list':
                    state.reset()
                    state.list_items = [x.name for x in expr.items]
                    state.list_vars = [inst.target.name]
                    state.stmts.append(inst)
                    continue
                elif expr.op == 'call' and expr in calltypes:
                    if guard(inline_array, inst.target, expr, state.stmts, state.list_vars, state.dels):
                        state.modified = True
                        continue
        elif isinstance(inst, ir.Del):
            removed_var = inst.value
            if removed_var in state.list_items:
                state.dels.append(inst)
                continue
            elif removed_var in state.list_vars:
                state.dead_vars.append(removed_var)
                state.list_vars.remove(removed_var)
                state.stmts.append(inst)
                if state.list_vars == []:
                    body = []
                    for inst in state.stmts:
                        if isinstance(inst, ir.Assign) and inst.target.name in state.dead_vars or (isinstance(inst, ir.Del) and inst.value in state.dead_vars):
                            continue
                        body.append(inst)
                    state.stmts = body
                    state.dead_vars = []
                    state.modified = True
                    continue
        state.stmts.append(inst)
        if state.list_var_used(inst):
            state.reset()
    return state.stmts if state.modified else None