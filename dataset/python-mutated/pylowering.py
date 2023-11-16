"""
Lowering implementation for object mode.
"""
import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import ForbiddenConstruct, LoweringError, NumbaNotImplementedError
from numba.core.lowering import BaseLower
_unsupported_builtins = set([locals])
PYTHON_BINOPMAP = {operator.add: ('number_add', False), operator.sub: ('number_subtract', False), operator.mul: ('number_multiply', False), operator.truediv: ('number_truedivide', False), operator.floordiv: ('number_floordivide', False), operator.mod: ('number_remainder', False), operator.pow: ('number_power', False), operator.lshift: ('number_lshift', False), operator.rshift: ('number_rshift', False), operator.and_: ('number_and', False), operator.or_: ('number_or', False), operator.xor: ('number_xor', False), operator.iadd: ('number_add', True), operator.isub: ('number_subtract', True), operator.imul: ('number_multiply', True), operator.itruediv: ('number_truedivide', True), operator.ifloordiv: ('number_floordivide', True), operator.imod: ('number_remainder', True), operator.ipow: ('number_power', True), operator.ilshift: ('number_lshift', True), operator.irshift: ('number_rshift', True), operator.iand: ('number_and', True), operator.ior: ('number_or', True), operator.ixor: ('number_xor', True)}
PYTHON_BINOPMAP[operator.matmul] = ('number_matrix_multiply', False)
PYTHON_BINOPMAP[operator.imatmul] = ('number_matrix_multiply', True)
PYTHON_COMPAREOPMAP = {operator.eq: '==', operator.ne: '!=', operator.lt: '<', operator.le: '<=', operator.gt: '>', operator.ge: '>=', operator.is_: 'is', operator.is_not: 'is not', operator.contains: 'in'}

class PyLower(BaseLower):
    GeneratorLower = generators.PyGeneratorLower

    def init(self):
        if False:
            while True:
                i = 10
        self._frozen_strings = set()
        self._live_vars = set()

    def pre_lower(self):
        if False:
            return 10
        super(PyLower, self).pre_lower()
        self.init_pyapi()

    def post_lower(self):
        if False:
            return 10
        pass

    def pre_block(self, block):
        if False:
            for i in range(10):
                print('nop')
        self.init_vars(block)

    def lower_inst(self, inst):
        if False:
            while True:
                i = 10
        if isinstance(inst, ir.Assign):
            value = self.lower_assign(inst)
            self.storevar(value, inst.target.name)
        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.object_setitem(target, index, value)
            self.check_int_status(ok)
        elif isinstance(inst, ir.DelItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)
            ok = self.pyapi.object_delitem(target, index)
            self.check_int_status(ok)
        elif isinstance(inst, ir.SetAttr):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.object_setattr(target, self._freeze_string(inst.attr), value)
            self.check_int_status(ok)
        elif isinstance(inst, ir.DelAttr):
            target = self.loadvar(inst.target.name)
            ok = self.pyapi.object_delattr(target, self._freeze_string(inst.attr))
            self.check_int_status(ok)
        elif isinstance(inst, ir.StoreMap):
            dct = self.loadvar(inst.dct.name)
            key = self.loadvar(inst.key.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.dict_setitem(dct, key, value)
            self.check_int_status(ok)
        elif isinstance(inst, ir.Return):
            retval = self.loadvar(inst.value.name)
            if self.generator_info:
                self.pyapi.decref(retval)
                self.genlower.return_from_generator(self)
                return
            self.call_conv.return_value(self.builder, retval)
        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            if cond.type == llvmlite.ir.IntType(1):
                istrue = cond
            else:
                istrue = self.pyapi.object_istrue(cond)
            zero = llvmlite.ir.Constant(istrue.type, None)
            pred = self.builder.icmp_unsigned('!=', istrue, zero)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]
            self.builder.cbranch(pred, tr, fl)
        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)
        elif isinstance(inst, ir.Del):
            self.delvar(inst.value)
        elif isinstance(inst, ir.PopBlock):
            pass
        elif isinstance(inst, ir.Raise):
            if inst.exception is not None:
                exc = self.loadvar(inst.exception.name)
                self.incref(exc)
            else:
                exc = None
            self.pyapi.raise_object(exc)
            self.return_exception_raised()
        else:
            msg = f'{type(inst)}, {inst}'
            raise NumbaNotImplementedError(msg)

    @cached_property
    def _omitted_typobj(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a `OmittedArg` type instance as a LLVM value suitable for\n        testing at runtime.\n        '
        from numba.core.dispatcher import OmittedArg
        return self.pyapi.unserialize(self.pyapi.serialize_object(OmittedArg))

    def lower_assign(self, inst):
        if False:
            print('Hello World!')
        '\n        The returned object must have a new reference\n        '
        value = inst.value
        if isinstance(value, (ir.Const, ir.FreeVar)):
            return self.lower_const(value.value)
        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            self.incref(val)
            return val
        elif isinstance(value, ir.Expr):
            return self.lower_expr(value)
        elif isinstance(value, ir.Global):
            return self.lower_global(value.name, value.value)
        elif isinstance(value, ir.Yield):
            return self.lower_yield(value)
        elif isinstance(value, ir.Arg):
            param = self.func_ir.func_id.pysig.parameters.get(value.name)
            obj = self.fnargs[value.index]
            slot = cgutils.alloca_once_value(self.builder, obj)
            if param is not None and param.default is inspect.Parameter.empty:
                self.incref(obj)
                self.builder.store(obj, slot)
            else:
                typobj = self.pyapi.get_type(obj)
                is_omitted = self.builder.icmp_unsigned('==', typobj, self._omitted_typobj)
                with self.builder.if_else(is_omitted, likely=False) as (omitted, present):
                    with present:
                        self.incref(obj)
                        self.builder.store(obj, slot)
                    with omitted:
                        obj = self.pyapi.object_getattr_string(obj, 'value')
                        self.builder.store(obj, slot)
            return self.builder.load(slot)
        else:
            raise NotImplementedError(type(value), value)

    def lower_yield(self, inst):
        if False:
            i = 10
            return i + 15
        yp = self.generator_info.yield_points[inst.index]
        assert yp.inst is inst
        self.genlower.init_generator_state(self)
        y = generators.LowerYield(self, yp, yp.live_vars | yp.weak_live_vars)
        y.lower_yield_suspend()
        val = self.loadvar(inst.value.name)
        self.pyapi.incref(val)
        self.call_conv.return_value(self.builder, val)
        y.lower_yield_resume()
        return self.pyapi.make_none()

    def lower_binop(self, expr, op, inplace=False):
        if False:
            print('Hello World!')
        lhs = self.loadvar(expr.lhs.name)
        rhs = self.loadvar(expr.rhs.name)
        assert not isinstance(op, str)
        if op in PYTHON_BINOPMAP:
            (fname, inplace) = PYTHON_BINOPMAP[op]
            fn = getattr(self.pyapi, fname)
            res = fn(lhs, rhs, inplace=inplace)
        else:
            fn = PYTHON_COMPAREOPMAP.get(expr.fn, expr.fn)
            if fn == 'in':
                (lhs, rhs) = (rhs, lhs)
            res = self.pyapi.object_richcompare(lhs, rhs, fn)
        self.check_error(res)
        return res

    def lower_expr(self, expr):
        if False:
            print('Hello World!')
        if expr.op == 'binop':
            return self.lower_binop(expr, expr.fn, inplace=False)
        elif expr.op == 'inplace_binop':
            return self.lower_binop(expr, expr.fn, inplace=True)
        elif expr.op == 'unary':
            value = self.loadvar(expr.value.name)
            if expr.fn == operator.neg:
                res = self.pyapi.number_negative(value)
            elif expr.fn == operator.pos:
                res = self.pyapi.number_positive(value)
            elif expr.fn == operator.not_:
                res = self.pyapi.object_not(value)
                self.check_int_status(res)
                res = self.pyapi.bool_from_bool(res)
            elif expr.fn == operator.invert:
                res = self.pyapi.number_invert(value)
            else:
                raise NotImplementedError(expr)
            self.check_error(res)
            return res
        elif expr.op == 'call':
            argvals = [self.loadvar(a.name) for a in expr.args]
            fn = self.loadvar(expr.func.name)
            args = self.pyapi.tuple_pack(argvals)
            if expr.vararg:
                varargs = self.pyapi.sequence_tuple(self.loadvar(expr.vararg.name))
                new_args = self.pyapi.sequence_concat(args, varargs)
                self.decref(varargs)
                self.decref(args)
                args = new_args
            if not expr.kws:
                ret = self.pyapi.call(fn, args, None)
            else:
                keyvalues = [(k, self.loadvar(v.name)) for (k, v) in expr.kws]
                kws = self.pyapi.dict_pack(keyvalues)
                ret = self.pyapi.call(fn, args, kws)
                self.decref(kws)
            self.decref(args)
            self.check_error(ret)
            return ret
        elif expr.op == 'getattr':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getattr(obj, self._freeze_string(expr.attr))
            self.check_error(res)
            return res
        elif expr.op == 'build_tuple':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.tuple_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'build_list':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.list_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'build_map':
            res = self.pyapi.dict_new(expr.size)
            self.check_error(res)
            for (k, v) in expr.items:
                key = self.loadvar(k.name)
                value = self.loadvar(v.name)
                ok = self.pyapi.dict_setitem(res, key, value)
                self.check_int_status(ok)
            return res
        elif expr.op == 'build_set':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.set_new()
            self.check_error(res)
            for it in items:
                ok = self.pyapi.set_add(res, it)
                self.check_int_status(ok)
            return res
        elif expr.op == 'getiter':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getiter(obj)
            self.check_error(res)
            return res
        elif expr.op == 'iternext':
            iterobj = self.loadvar(expr.value.name)
            item = self.pyapi.iter_next(iterobj)
            is_valid = cgutils.is_not_null(self.builder, item)
            pair = self.pyapi.tuple_new(2)
            with self.builder.if_else(is_valid) as (then, otherwise):
                with then:
                    self.pyapi.tuple_setitem(pair, 0, item)
                with otherwise:
                    self.check_occurred()
                    self.pyapi.tuple_setitem(pair, 0, self.pyapi.make_none())
            self.pyapi.tuple_setitem(pair, 1, self.pyapi.bool_from_bool(is_valid))
            return pair
        elif expr.op == 'pair_first':
            pair = self.loadvar(expr.value.name)
            first = self.pyapi.tuple_getitem(pair, 0)
            self.incref(first)
            return first
        elif expr.op == 'pair_second':
            pair = self.loadvar(expr.value.name)
            second = self.pyapi.tuple_getitem(pair, 1)
            self.incref(second)
            return second
        elif expr.op == 'exhaust_iter':
            iterobj = self.loadvar(expr.value.name)
            tup = self.pyapi.sequence_tuple(iterobj)
            self.check_error(tup)
            tup_size = self.pyapi.tuple_size(tup)
            expected_size = self.context.get_constant(types.intp, expr.count)
            has_wrong_size = self.builder.icmp_unsigned('!=', tup_size, expected_size)
            with cgutils.if_unlikely(self.builder, has_wrong_size):
                self.return_exception(ValueError)
            return tup
        elif expr.op == 'getitem':
            value = self.loadvar(expr.value.name)
            index = self.loadvar(expr.index.name)
            res = self.pyapi.object_getitem(value, index)
            self.check_error(res)
            return res
        elif expr.op == 'static_getitem':
            value = self.loadvar(expr.value.name)
            index = self.context.get_constant(types.intp, expr.index)
            indexobj = self.pyapi.long_from_ssize_t(index)
            self.check_error(indexobj)
            res = self.pyapi.object_getitem(value, indexobj)
            self.decref(indexobj)
            self.check_error(res)
            return res
        elif expr.op == 'getslice':
            target = self.loadvar(expr.target.name)
            start = self.loadvar(expr.start.name)
            stop = self.loadvar(expr.stop.name)
            slicefn = self.get_builtin_obj('slice')
            sliceobj = self.pyapi.call_function_objargs(slicefn, (start, stop))
            self.decref(slicefn)
            self.check_error(sliceobj)
            res = self.pyapi.object_getitem(target, sliceobj)
            self.check_error(res)
            return res
        elif expr.op == 'cast':
            val = self.loadvar(expr.value.name)
            self.incref(val)
            return val
        elif expr.op == 'phi':
            raise LoweringError('PHI not stripped')
        elif expr.op == 'null':
            return cgutils.get_null_value(self.pyapi.pyobj)
        else:
            raise NotImplementedError(expr)

    def lower_const(self, const):
        if False:
            print('Hello World!')
        index = self.env_manager.add_const(const)
        ret = self.env_manager.read_const(index)
        self.check_error(ret)
        self.incref(ret)
        return ret

    def lower_global(self, name, value):
        if False:
            return 10
        '\n        1) Check global scope dictionary.\n        2) Check __builtins__.\n            2a) is it a dictionary (for non __main__ module)\n            2b) is it a module (for __main__ module)\n        '
        moddict = self.get_module_dict()
        obj = self.pyapi.dict_getitem(moddict, self._freeze_string(name))
        self.incref(obj)
        try:
            if value in _unsupported_builtins:
                raise ForbiddenConstruct('builtins %s() is not supported' % name, loc=self.loc)
        except TypeError:
            pass
        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block
            with self.builder.if_then(obj_is_null):
                mod = self.pyapi.dict_getitem(moddict, self._freeze_string('__builtins__'))
                builtin = self.builtin_lookup(mod, name)
                bbif = self.builder.basic_block
            retval = self.builder.phi(self.pyapi.pyobj)
            retval.add_incoming(obj, bbelse)
            retval.add_incoming(builtin, bbif)
        else:
            retval = obj
            with cgutils.if_unlikely(self.builder, self.is_null(retval)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()
        return retval

    def get_module_dict(self):
        if False:
            while True:
                i = 10
        return self.env_body.globals

    def get_builtin_obj(self, name):
        if False:
            while True:
                i = 10
        moddict = self.get_module_dict()
        mod = self.pyapi.dict_getitem(moddict, self._freeze_string('__builtins__'))
        return self.builtin_lookup(mod, name)

    def builtin_lookup(self, mod, name):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args\n        ----\n        mod:\n            The __builtins__ dictionary or module, as looked up in\n            a module's globals.\n        name: str\n            The object to lookup\n        "
        fromdict = self.pyapi.dict_getitem(mod, self._freeze_string(name))
        self.incref(fromdict)
        bbifdict = self.builder.basic_block
        with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
            frommod = self.pyapi.object_getattr(mod, self._freeze_string(name))
            with cgutils.if_unlikely(self.builder, self.is_null(frommod)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()
            bbifmod = self.builder.basic_block
        builtin = self.builder.phi(self.pyapi.pyobj)
        builtin.add_incoming(fromdict, bbifdict)
        builtin.add_incoming(frommod, bbifmod)
        return builtin

    def check_occurred(self):
        if False:
            print('Hello World!')
        '\n        Return if an exception occurred.\n        '
        err_occurred = cgutils.is_not_null(self.builder, self.pyapi.err_occurred())
        with cgutils.if_unlikely(self.builder, err_occurred):
            self.return_exception_raised()

    def check_error(self, obj):
        if False:
            return 10
        '\n        Return if *obj* is NULL.\n        '
        with cgutils.if_unlikely(self.builder, self.is_null(obj)):
            self.return_exception_raised()
        return obj

    def check_int_status(self, num, ok_value=0):
        if False:
            i = 10
            return i + 15
        '\n        Raise an exception if *num* is smaller than *ok_value*.\n        '
        ok = llvmlite.ir.Constant(num.type, ok_value)
        pred = self.builder.icmp_signed('<', num, ok)
        with cgutils.if_unlikely(self.builder, pred):
            self.return_exception_raised()

    def is_null(self, obj):
        if False:
            print('Hello World!')
        return cgutils.is_null(self.builder, obj)

    def return_exception_raised(self):
        if False:
            while True:
                i = 10
        '\n        Return with the currently raised exception.\n        '
        self.cleanup_vars()
        self.call_conv.return_exc(self.builder)

    def init_vars(self, block):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize live variables for *block*.\n        '
        self._live_vars = set(self.func_ir.get_block_entry_vars(block))

    def _getvar(self, name, ltype=None):
        if False:
            print('Hello World!')
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, ltype=ltype)
        return self.varmap[name]

    def loadvar(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load the llvm value of the variable named *name*.\n        '
        assert name in self._live_vars, name
        ptr = self.varmap[name]
        val = self.builder.load(ptr)
        with cgutils.if_unlikely(self.builder, self.is_null(val)):
            self.pyapi.raise_missing_name_error(name)
            self.return_exception_raised()
        return val

    def delvar(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the variable slot with the given name. This will decref\n        the corresponding Python object.\n        '
        self._live_vars.remove(name)
        ptr = self._getvar(name)
        self.decref(self.builder.load(ptr))
        self.builder.store(cgutils.get_null_value(ptr.type.pointee), ptr)

    def storevar(self, value, name, clobber=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stores a llvm value and allocate stack slot if necessary.\n        The llvm value can be of arbitrary type.\n        '
        is_redefine = name in self._live_vars and (not clobber)
        ptr = self._getvar(name, ltype=value.type)
        if is_redefine:
            old = self.builder.load(ptr)
        else:
            self._live_vars.add(name)
        assert value.type == ptr.type.pointee, (str(value.type), str(ptr.type.pointee))
        self.builder.store(value, ptr)
        if is_redefine:
            self.decref(old)

    def cleanup_vars(self):
        if False:
            i = 10
            return i + 15
        '\n        Cleanup live variables.\n        '
        for name in self._live_vars:
            ptr = self._getvar(name)
            self.decref(self.builder.load(ptr))

    def alloca(self, name, ltype=None):
        if False:
            return 10
        '\n        Allocate a stack slot and initialize it to NULL.\n        The default is to allocate a pyobject pointer.\n        Use ``ltype`` to override.\n        '
        if ltype is None:
            ltype = self.context.get_value_type(types.pyobject)
        with self.builder.goto_block(self.entry_block):
            ptr = self.builder.alloca(ltype, name=name)
            self.builder.store(cgutils.get_null_value(ltype), ptr)
        return ptr

    def _alloca_var(self, name, fetype):
        if False:
            for i in range(10):
                print('nop')
        return self.alloca(name)

    def incref(self, value):
        if False:
            while True:
                i = 10
        self.pyapi.incref(value)

    def decref(self, value):
        if False:
            print('Hello World!')
        '\n        This is allow to be called on non pyobject pointer, in which case\n        no code is inserted.\n        '
        lpyobj = self.context.get_value_type(types.pyobject)
        if value.type == lpyobj:
            self.pyapi.decref(value)

    def _freeze_string(self, string):
        if False:
            for i in range(10):
                print('nop')
        '\n        Freeze a Python string object into the code.\n        '
        return self.lower_const(string)