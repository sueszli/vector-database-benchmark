import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import CallableTemplate, signature, infer_global, AbstractTemplate
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support

class StencilFuncLowerer(object):
    """Callable class responsible for lowering calls to a specific StencilFunc.
    """

    def __init__(self, sf):
        if False:
            while True:
                i = 10
        self.stencilFunc = sf

    def __call__(self, context, builder, sig, args):
        if False:
            while True:
                i = 10
        cres = self.stencilFunc.compile_for_argtys(sig.args, {}, sig.return_type, None)
        res = context.call_internal(builder, cres.fndesc, sig, args)
        context.add_linking_libs([cres.library])
        return res

@register_jitable
def raise_if_incompatible_array_sizes(a, *args):
    if False:
        for i in range(10):
            print('nop')
    ashape = a.shape
    for arg in literal_unroll(args):
        if a.ndim != arg.ndim:
            raise ValueError('Secondary stencil array does not have same number  of dimensions as the first stencil input.')
        argshape = arg.shape
        for i in range(len(ashape)):
            if ashape[i] > argshape[i]:
                raise ValueError('Secondary stencil array has some dimension smaller the same dimension in the first stencil input.')

def slice_addition(the_slice, addend):
    if False:
        print('Hello World!')
    ' Called by stencil in Python mode to add the loop index to a\n        user-specified slice.\n    '
    return slice(the_slice.start + addend, the_slice.stop + addend)

class StencilFunc(object):
    """
    A special type to hold stencil information for the IR.
    """
    id_counter = 0

    def __init__(self, kernel_ir, mode, options):
        if False:
            print('Hello World!')
        self.id = type(self).id_counter
        type(self).id_counter += 1
        self.kernel_ir = kernel_ir
        self.mode = mode
        self.options = options
        self.kws = []
        self._typingctx = registry.cpu_target.typing_context
        self._targetctx = registry.cpu_target.target_context
        self._typingctx.refresh()
        self._targetctx.refresh()
        self._install_type(self._typingctx)
        self.neighborhood = self.options.get('neighborhood')
        self._type_cache = {}
        self._lower_me = StencilFuncLowerer(self)

    def replace_return_with_setitem(self, blocks, index_vars, out_name):
        if False:
            i = 10
            return i + 15
        '\n        Find return statements in the IR and replace them with a SetItem\n        call of the value "returned" by the kernel into the result array.\n        Returns the block labels that contained return statements.\n        '
        ret_blocks = []
        for (label, block) in blocks.items():
            scope = block.scope
            loc = block.loc
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Return):
                    ret_blocks.append(label)
                    if len(index_vars) == 1:
                        rvar = ir.Var(scope, out_name, loc)
                        ivar = ir.Var(scope, index_vars[0], loc)
                        new_body.append(ir.SetItem(rvar, ivar, stmt.value, loc))
                    else:
                        var_index_vars = []
                        for one_var in index_vars:
                            index_var = ir.Var(scope, one_var, loc)
                            var_index_vars += [index_var]
                        s_index_var = scope.redefine('stencil_index', loc)
                        tuple_call = ir.Expr.build_tuple(var_index_vars, loc)
                        new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                        rvar = ir.Var(scope, out_name, loc)
                        si = ir.SetItem(rvar, s_index_var, stmt.value, loc)
                        new_body.append(si)
                else:
                    new_body.append(stmt)
            block.body = new_body
        return ret_blocks

    def add_indices_to_kernel(self, kernel, index_names, ndim, neighborhood, standard_indexed, typemap, calltypes):
        if False:
            return 10
        "\n        Transforms the stencil kernel as specified by the user into one\n        that includes each dimension's index variable as part of the getitem\n        calls.  So, in effect array[-1] becomes array[index0-1].\n        "
        const_dict = {}
        kernel_consts = []
        if config.DEBUG_ARRAY_OPT >= 1:
            print('add_indices_to_kernel', ndim, neighborhood)
            ir_utils.dump_blocks(kernel.blocks)
        if neighborhood is None:
            need_to_calc_kernel = True
        else:
            need_to_calc_kernel = False
            if len(neighborhood) != ndim:
                raise ValueError('%d dimensional neighborhood specified for %d dimensional input array' % (len(neighborhood), ndim))
        tuple_table = ir_utils.get_tuple_table(kernel.blocks)
        relatively_indexed = set()
        for block in kernel.blocks.values():
            scope = block.scope
            loc = block.loc
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Const):
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print('remembering in const_dict', stmt.target.name, stmt.value.value)
                    const_dict[stmt.target.name] = stmt.value.value
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['setitem', 'static_setitem']) and (stmt.value.value.name in kernel.arg_names) or (isinstance(stmt, ir.SetItem) and stmt.target.name in kernel.arg_names):
                    raise ValueError('Assignments to arrays passed to stencil kernels is not allowed.')
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['getitem', 'static_getitem']) and (stmt.value.value.name in kernel.arg_names) and (stmt.value.value.name not in standard_indexed):
                    if stmt.value.op == 'getitem':
                        stmt_index_var = stmt.value.index
                    else:
                        stmt_index_var = stmt.value.index_var
                    relatively_indexed.add(stmt.value.value.name)
                    if need_to_calc_kernel:
                        assert hasattr(stmt_index_var, 'name')
                        if stmt_index_var.name in tuple_table:
                            kernel_consts += [tuple_table[stmt_index_var.name]]
                        elif stmt_index_var.name in const_dict:
                            kernel_consts += [const_dict[stmt_index_var.name]]
                        else:
                            raise NumbaValueError("stencil kernel index is not constant, 'neighborhood' option required")
                    if ndim == 1:
                        index_var = ir.Var(scope, index_names[0], loc)
                        tmpvar = scope.redefine('stencil_index', loc)
                        stmt_index_var_typ = typemap[stmt_index_var.name]
                        if isinstance(stmt_index_var_typ, types.misc.SliceType):
                            sa_var = scope.redefine('slice_addition', loc)
                            sa_func = numba.njit(slice_addition)
                            sa_func_typ = types.functions.Dispatcher(sa_func)
                            typemap[sa_var.name] = sa_func_typ
                            g_sa = ir.Global('slice_addition', sa_func, loc)
                            new_body.append(ir.Assign(g_sa, sa_var, loc))
                            slice_addition_call = ir.Expr.call(sa_var, [stmt_index_var, index_var], (), loc)
                            calltypes[slice_addition_call] = sa_func_typ.get_call_type(self._typingctx, [stmt_index_var_typ, types.intp], {})
                            new_body.append(ir.Assign(slice_addition_call, tmpvar, loc))
                            new_body.append(ir.Assign(ir.Expr.getitem(stmt.value.value, tmpvar, loc), stmt.target, loc))
                        else:
                            acc_call = ir.Expr.binop(operator.add, stmt_index_var, index_var, loc)
                            new_body.append(ir.Assign(acc_call, tmpvar, loc))
                            new_body.append(ir.Assign(ir.Expr.getitem(stmt.value.value, tmpvar, loc), stmt.target, loc))
                    else:
                        index_vars = []
                        sum_results = []
                        s_index_var = scope.redefine('stencil_index', loc)
                        const_index_vars = []
                        ind_stencils = []
                        stmt_index_var_typ = typemap[stmt_index_var.name]
                        for dim in range(ndim):
                            tmpvar = scope.redefine('const_index', loc)
                            new_body.append(ir.Assign(ir.Const(dim, loc), tmpvar, loc))
                            const_index_vars += [tmpvar]
                            index_var = ir.Var(scope, index_names[dim], loc)
                            index_vars += [index_var]
                            tmpvar = scope.redefine('ind_stencil_index', loc)
                            ind_stencils += [tmpvar]
                            getitemvar = scope.redefine('getitem', loc)
                            getitemcall = ir.Expr.getitem(stmt_index_var, const_index_vars[dim], loc)
                            new_body.append(ir.Assign(getitemcall, getitemvar, loc))
                            if isinstance(stmt_index_var_typ, types.ConstSized):
                                one_index_typ = stmt_index_var_typ[dim]
                            else:
                                one_index_typ = stmt_index_var_typ[:]
                            if isinstance(one_index_typ, types.misc.SliceType):
                                sa_var = scope.redefine('slice_addition', loc)
                                sa_func = numba.njit(slice_addition)
                                sa_func_typ = types.functions.Dispatcher(sa_func)
                                typemap[sa_var.name] = sa_func_typ
                                g_sa = ir.Global('slice_addition', sa_func, loc)
                                new_body.append(ir.Assign(g_sa, sa_var, loc))
                                slice_addition_call = ir.Expr.call(sa_var, [getitemvar, index_vars[dim]], (), loc)
                                calltypes[slice_addition_call] = sa_func_typ.get_call_type(self._typingctx, [one_index_typ, types.intp], {})
                                new_body.append(ir.Assign(slice_addition_call, tmpvar, loc))
                            else:
                                acc_call = ir.Expr.binop(operator.add, getitemvar, index_vars[dim], loc)
                                new_body.append(ir.Assign(acc_call, tmpvar, loc))
                        tuple_call = ir.Expr.build_tuple(ind_stencils, loc)
                        new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                        new_body.append(ir.Assign(ir.Expr.getitem(stmt.value.value, s_index_var, loc), stmt.target, loc))
                else:
                    new_body.append(stmt)
            block.body = new_body
        if need_to_calc_kernel:
            neighborhood = [[0, 0] for _ in range(ndim)]
            if len(kernel_consts) == 0:
                raise NumbaValueError('Stencil kernel with no accesses to relatively indexed arrays.')
            for index in kernel_consts:
                if isinstance(index, tuple) or isinstance(index, list):
                    for i in range(len(index)):
                        te = index[i]
                        if isinstance(te, ir.Var) and te.name in const_dict:
                            te = const_dict[te.name]
                        if isinstance(te, int):
                            neighborhood[i][0] = min(neighborhood[i][0], te)
                            neighborhood[i][1] = max(neighborhood[i][1], te)
                        else:
                            raise NumbaValueError("stencil kernel index is not constant,'neighborhood' option required")
                    index_len = len(index)
                elif isinstance(index, int):
                    neighborhood[0][0] = min(neighborhood[0][0], index)
                    neighborhood[0][1] = max(neighborhood[0][1], index)
                    index_len = 1
                else:
                    raise NumbaValueError('Non-tuple or non-integer used as stencil index.')
                if index_len != ndim:
                    raise NumbaValueError('Stencil index does not match array dimensionality.')
        return (neighborhood, relatively_indexed)

    def get_return_type(self, argtys):
        if False:
            i = 10
            return i + 15
        if config.DEBUG_ARRAY_OPT >= 1:
            print('get_return_type', argtys)
            ir_utils.dump_blocks(self.kernel_ir.blocks)
        if not isinstance(argtys[0], types.npytypes.Array):
            raise NumbaValueError('The first argument to a stencil kernel must be the primary input array.')
        from numba.core import typed_passes
        (typemap, return_type, calltypes, _) = typed_passes.type_inference_stage(self._typingctx, self._targetctx, self.kernel_ir, argtys, None, {})
        if isinstance(return_type, types.npytypes.Array):
            raise NumbaValueError('Stencil kernel must return a scalar and not a numpy array.')
        real_ret = types.npytypes.Array(return_type, argtys[0].ndim, argtys[0].layout)
        return (real_ret, typemap, calltypes)

    def _install_type(self, typingctx):
        if False:
            print('Hello World!')
        'Constructs and installs a typing class for a StencilFunc object in\n        the input typing context.\n        '
        _ty_cls = type('StencilFuncTyping_' + str(self.id), (AbstractTemplate,), dict(key=self, generic=self._type_me))
        typingctx.insert_user_function(self, _ty_cls)

    def compile_for_argtys(self, argtys, kwtys, return_type, sigret):
        if False:
            for i in range(10):
                print('nop')
        (_, result, typemap, calltypes) = self._type_cache[argtys]
        new_func = self._stencil_wrapper(result, sigret, return_type, typemap, calltypes, *argtys)
        return new_func

    def _type_me(self, argtys, kwtys):
        if False:
            while True:
                i = 10
        '\n        Implement AbstractTemplate.generic() for the typing class\n        built by StencilFunc._install_type().\n        Return the call-site signature.\n        '
        if self.neighborhood is not None and len(self.neighborhood) != argtys[0].ndim:
            raise NumbaValueError('%d dimensional neighborhood specified for %d dimensional input array' % (len(self.neighborhood), argtys[0].ndim))
        argtys_extra = argtys
        sig_extra = ''
        result = None
        if 'out' in kwtys:
            argtys_extra += (kwtys['out'],)
            sig_extra += ', out=None'
            result = kwtys['out']
        if 'neighborhood' in kwtys:
            argtys_extra += (kwtys['neighborhood'],)
            sig_extra += ', neighborhood=None'
        if argtys_extra in self._type_cache:
            (_sig, _, _, _) = self._type_cache[argtys_extra]
            return _sig
        (real_ret, typemap, calltypes) = self.get_return_type(argtys)
        sig = signature(real_ret, *argtys_extra)
        dummy_text = 'def __numba_dummy_stencil({}{}):\n    pass\n'.format(','.join(self.kernel_ir.arg_names), sig_extra)
        (exec(dummy_text) in globals(), locals())
        dummy_func = eval('__numba_dummy_stencil')
        sig = sig.replace(pysig=utils.pysignature(dummy_func))
        self._targetctx.insert_func_defn([(self._lower_me, self, argtys_extra)])
        self._type_cache[argtys_extra] = (sig, result, typemap, calltypes)
        return sig

    def copy_ir_with_calltypes(self, ir, calltypes):
        if False:
            return 10
        '\n        Create a copy of a given IR along with its calltype information.\n        We need a copy of the calltypes because copy propagation applied\n        to the copied IR will change the calltypes and make subsequent\n        uses of the original IR invalid.\n        '
        copy_calltypes = {}
        kernel_copy = ir.copy()
        kernel_copy.blocks = {}
        for (block_label, block) in ir.blocks.items():
            new_block = copy.deepcopy(ir.blocks[block_label])
            new_block.body = []
            for stmt in ir.blocks[block_label].body:
                scopy = copy.deepcopy(stmt)
                new_block.body.append(scopy)
                if stmt in calltypes:
                    copy_calltypes[scopy] = calltypes[stmt]
            kernel_copy.blocks[block_label] = new_block
        return (kernel_copy, copy_calltypes)

    def _stencil_wrapper(self, result, sigret, return_type, typemap, calltypes, *args):
        if False:
            while True:
                i = 10
        (kernel_copy, copy_calltypes) = self.copy_ir_with_calltypes(self.kernel_ir, calltypes)
        ir_utils.remove_args(kernel_copy.blocks)
        first_arg = kernel_copy.arg_names[0]
        (in_cps, out_cps) = ir_utils.copy_propagate(kernel_copy.blocks, typemap)
        name_var_table = ir_utils.get_name_var_table(kernel_copy.blocks)
        ir_utils.apply_copy_propagate(kernel_copy.blocks, in_cps, name_var_table, typemap, copy_calltypes)
        if 'out' in name_var_table:
            raise NumbaValueError("Cannot use the reserved word 'out' in stencil kernels.")
        sentinel_name = ir_utils.get_unused_var_name('__sentinel__', name_var_table)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('name_var_table', name_var_table, sentinel_name)
        the_array = args[0]
        if config.DEBUG_ARRAY_OPT >= 1:
            print('_stencil_wrapper', return_type, return_type.dtype, type(return_type.dtype), args)
            ir_utils.dump_blocks(kernel_copy.blocks)
        stencil_func_name = '__numba_stencil_%s_%s' % (hex(id(the_array)).replace('-', '_'), self.id)
        index_vars = []
        for i in range(the_array.ndim):
            index_var_name = ir_utils.get_unused_var_name('index' + str(i), name_var_table)
            index_vars += [index_var_name]
        out_name = ir_utils.get_unused_var_name('out', name_var_table)
        neighborhood_name = ir_utils.get_unused_var_name('neighborhood', name_var_table)
        sig_extra = ''
        if result is not None:
            sig_extra += ', {}=None'.format(out_name)
        if 'neighborhood' in dict(self.kws):
            sig_extra += ', {}=None'.format(neighborhood_name)
        standard_indexed = self.options.get('standard_indexing', [])
        if first_arg in standard_indexed:
            raise NumbaValueError('The first argument to a stencil kernel must use relative indexing, not standard indexing.')
        if len(set(standard_indexed) - set(kernel_copy.arg_names)) != 0:
            raise NumbaValueError('Standard indexing requested for an array name not present in the stencil kernel definition.')
        (kernel_size, relatively_indexed) = self.add_indices_to_kernel(kernel_copy, index_vars, the_array.ndim, self.neighborhood, standard_indexed, typemap, copy_calltypes)
        if self.neighborhood is None:
            self.neighborhood = kernel_size
        if config.DEBUG_ARRAY_OPT >= 1:
            print('After add_indices_to_kernel')
            ir_utils.dump_blocks(kernel_copy.blocks)
        ret_blocks = self.replace_return_with_setitem(kernel_copy.blocks, index_vars, out_name)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('After replace_return_with_setitem', ret_blocks)
            ir_utils.dump_blocks(kernel_copy.blocks)
        func_text = 'def {}({}{}):\n'.format(stencil_func_name, ','.join(kernel_copy.arg_names), sig_extra)
        ranges = []
        for i in range(the_array.ndim):
            if isinstance(kernel_size[i][0], int):
                lo = kernel_size[i][0]
                hi = kernel_size[i][1]
            else:
                lo = '{}[{}][0]'.format(neighborhood_name, i)
                hi = '{}[{}][1]'.format(neighborhood_name, i)
            ranges.append((lo, hi))
        if len(relatively_indexed) > 1:
            func_text += '    raise_if_incompatible_array_sizes(' + first_arg
            for other_array in relatively_indexed:
                if other_array != first_arg:
                    func_text += ',' + other_array
            func_text += ')\n'
        shape_name = ir_utils.get_unused_var_name('full_shape', name_var_table)
        func_text += '    {} = {}.shape\n'.format(shape_name, first_arg)

        def cval_as_str(cval):
            if False:
                i = 10
                return i + 15
            if not np.isfinite(cval):
                if np.isnan(cval):
                    return 'np.nan'
                elif np.isinf(cval):
                    if cval < 0:
                        return '-np.inf'
                    else:
                        return 'np.inf'
            else:
                return str(cval)
        if result is None:
            return_type_name = numpy_support.as_dtype(return_type.dtype).type.__name__
            out_init = '{} = np.empty({}, dtype=np.{})\n'.format(out_name, shape_name, return_type_name)
            if 'cval' in self.options:
                cval = self.options['cval']
                cval_ty = typing.typeof.typeof(cval)
                if not self._typingctx.can_convert(cval_ty, return_type.dtype):
                    msg = 'cval type does not match stencil return type.'
                    raise NumbaValueError(msg)
            else:
                cval = 0
            func_text += '    ' + out_init
            for dim in range(the_array.ndim):
                start_items = [':'] * the_array.ndim
                end_items = [':'] * the_array.ndim
                start_items[dim] = ':-{}'.format(self.neighborhood[dim][0])
                end_items[dim] = '-{}:'.format(self.neighborhood[dim][1])
                func_text += '    ' + '{}[{}] = {}\n'.format(out_name, ','.join(start_items), cval_as_str(cval))
                func_text += '    ' + '{}[{}] = {}\n'.format(out_name, ','.join(end_items), cval_as_str(cval))
        elif 'cval' in self.options:
            cval = self.options['cval']
            cval_ty = typing.typeof.typeof(cval)
            if not self._typingctx.can_convert(cval_ty, return_type.dtype):
                msg = 'cval type does not match stencil return type.'
                raise NumbaValueError(msg)
            out_init = '{}[:] = {}\n'.format(out_name, cval_as_str(cval))
            func_text += '    ' + out_init
        offset = 1
        for i in range(the_array.ndim):
            for j in range(offset):
                func_text += '    '
            func_text += 'for {} in range(-min(0,{}),{}[{}]-max(0,{})):\n'.format(index_vars[i], ranges[i][0], shape_name, i, ranges[i][1])
            offset += 1
        for j in range(offset):
            func_text += '    '
        func_text += '{} = 0\n'.format(sentinel_name)
        func_text += '    return {}\n'.format(out_name)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('new stencil func text')
            print(func_text)
        (exec(func_text) in globals(), locals())
        stencil_func = eval(stencil_func_name)
        if sigret is not None:
            pysig = utils.pysignature(stencil_func)
            sigret.pysig = pysig
        from numba.core import compiler
        stencil_ir = compiler.run_frontend(stencil_func)
        ir_utils.remove_dels(stencil_ir.blocks)
        var_table = ir_utils.get_name_var_table(stencil_ir.blocks)
        new_var_dict = {}
        reserved_names = [sentinel_name, out_name, neighborhood_name, shape_name] + kernel_copy.arg_names + index_vars
        for (name, var) in var_table.items():
            if not name in reserved_names:
                assert isinstance(var, ir.Var)
                new_var = var.scope.redefine(var.name, var.loc)
                new_var_dict[name] = new_var.name
        ir_utils.replace_var_names(stencil_ir.blocks, new_var_dict)
        stencil_stub_last_label = max(stencil_ir.blocks.keys()) + 1
        kernel_copy.blocks = ir_utils.add_offset_to_labels(kernel_copy.blocks, stencil_stub_last_label)
        new_label = max(kernel_copy.blocks.keys()) + 1
        ret_blocks = [x + stencil_stub_last_label for x in ret_blocks]
        if config.DEBUG_ARRAY_OPT >= 1:
            print('ret_blocks w/ offsets', ret_blocks, stencil_stub_last_label)
            print('before replace sentinel stencil_ir')
            ir_utils.dump_blocks(stencil_ir.blocks)
            print('before replace sentinel kernel_copy')
            ir_utils.dump_blocks(kernel_copy.blocks)
        for (label, block) in stencil_ir.blocks.items():
            for (i, inst) in enumerate(block.body):
                if isinstance(inst, ir.Assign) and inst.target.name == sentinel_name:
                    loc = inst.loc
                    scope = block.scope
                    prev_block = ir.Block(scope, loc)
                    prev_block.body = block.body[:i]
                    block.body = block.body[i + 1:]
                    body_first_label = min(kernel_copy.blocks.keys())
                    prev_block.append(ir.Jump(body_first_label, loc))
                    for (l, b) in kernel_copy.blocks.items():
                        stencil_ir.blocks[l] = b
                    stencil_ir.blocks[new_label] = block
                    stencil_ir.blocks[label] = prev_block
                    for ret_block in ret_blocks:
                        stencil_ir.blocks[ret_block].append(ir.Jump(new_label, loc))
                    break
            else:
                continue
            break
        stencil_ir.blocks = ir_utils.rename_labels(stencil_ir.blocks)
        ir_utils.remove_dels(stencil_ir.blocks)
        assert isinstance(the_array, types.Type)
        array_types = args
        new_stencil_param_types = list(array_types)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('new_stencil_param_types', new_stencil_param_types)
            ir_utils.dump_blocks(stencil_ir.blocks)
        ir_utils.fixup_var_define_in_scope(stencil_ir.blocks)
        new_func = compiler.compile_ir(self._typingctx, self._targetctx, stencil_ir, new_stencil_param_types, None, compiler.DEFAULT_FLAGS, {})
        return new_func

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.neighborhood is not None and len(self.neighborhood) != args[0].ndim:
            raise ValueError('{} dimensional neighborhood specified for {} dimensional input array'.format(len(self.neighborhood), args[0].ndim))
        if 'out' in kwargs:
            result = kwargs['out']
            rdtype = result.dtype
            rttype = numpy_support.from_dtype(rdtype)
            result_type = types.npytypes.Array(rttype, result.ndim, numpy_support.map_layout(result))
            array_types = tuple([typing.typeof.typeof(x) for x in args])
            array_types_full = tuple([typing.typeof.typeof(x) for x in args] + [result_type])
        else:
            result = None
            array_types = tuple([typing.typeof.typeof(x) for x in args])
            array_types_full = array_types
        if config.DEBUG_ARRAY_OPT >= 1:
            print('__call__', array_types, args, kwargs)
        (real_ret, typemap, calltypes) = self.get_return_type(array_types)
        new_func = self._stencil_wrapper(result, None, real_ret, typemap, calltypes, *array_types_full)
        if result is None:
            return new_func.entry_point(*args)
        else:
            return new_func.entry_point(*args + (result,))

def stencil(func_or_mode='constant', **options):
    if False:
        i = 10
        return i + 15
    if not isinstance(func_or_mode, str):
        mode = 'constant'
        func = func_or_mode
    else:
        mode = func_or_mode
        func = None
    for option in options:
        if option not in ['cval', 'standard_indexing', 'neighborhood']:
            raise ValueError('Unknown stencil option ' + option)
    wrapper = _stencil(mode, options)
    if func is not None:
        return wrapper(func)
    return wrapper

def _stencil(mode, options):
    if False:
        for i in range(10):
            print('nop')
    if mode != 'constant':
        raise ValueError('Unsupported mode style ' + mode)

    def decorated(func):
        if False:
            return 10
        from numba.core import compiler
        kernel_ir = compiler.run_frontend(func)
        return StencilFunc(kernel_ir, mode, options)
    return decorated

@lower_builtin(stencil)
def stencil_dummy_lower(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    'lowering for dummy stencil calls'
    return lir.Constant(lir.IntType(types.intp.bitwidth), 0)