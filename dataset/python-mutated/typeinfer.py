"""
Type inference base on CPA.
The algorithm guarantees monotonic growth of type-sets for each variable.

Steps:
    1. seed initial types
    2. build constraints
    3. propagate constraints
    4. unify types

Constraint propagation is precise and does not regret (no backtracing).
Constraints push types forward following the dataflow.
"""
import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import TypingError, UntypedAttributeError, new_error_context, termcolor, UnsupportedError, ForceLiteralArg, CompilerError, NumbaValueError
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
_logger = logging.getLogger(__name__)

class NOTSET:
    pass
_termcolor = termcolor()

class TypeVar(object):

    def __init__(self, context, var):
        if False:
            return 10
        self.context = context
        self.var = var
        self.type = None
        self.locked = False
        self.define_loc = None
        self.literal_value = NOTSET

    def add_type(self, tp, loc):
        if False:
            return 10
        assert isinstance(tp, types.Type), type(tp)
        if self.locked:
            if tp != self.type:
                if self.context.can_convert(tp, self.type) is None:
                    msg = "No conversion from %s to %s for '%s', defined at %s"
                    raise TypingError(msg % (tp, self.type, self.var, self.define_loc), loc=loc)
        else:
            if self.type is not None:
                unified = self.context.unify_pairs(self.type, tp)
                if unified is None:
                    msg = "Cannot unify %s and %s for '%s', defined at %s"
                    raise TypingError(msg % (self.type, tp, self.var, self.define_loc), loc=self.define_loc)
            else:
                unified = tp
                self.define_loc = loc
            self.type = unified
        return self.type

    def lock(self, tp, loc, literal_value=NOTSET):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(tp, types.Type), type(tp)
        if self.locked:
            msg = 'Invalid reassignment of a type-variable detected, type variables are locked according to the user provided function signature or from an ir.Const node. This is a bug! Type={}. {}'.format(tp, self.type)
            raise CompilerError(msg, loc)
        if self.type is not None and self.context.can_convert(self.type, tp) is None:
            raise TypingError("No conversion from %s to %s for '%s'" % (tp, self.type, self.var), loc=loc)
        self.type = tp
        self.locked = True
        if self.define_loc is None:
            self.define_loc = loc
        self.literal_value = literal_value

    def union(self, other, loc):
        if False:
            print('Hello World!')
        if other.type is not None:
            self.add_type(other.type, loc=loc)
        return self.type

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s := %s' % (self.var, self.type or '<undecided>')

    @property
    def defined(self):
        if False:
            i = 10
            return i + 15
        return self.type is not None

    def get(self):
        if False:
            print('Hello World!')
        return (self.type,) if self.type is not None else ()

    def getone(self):
        if False:
            print('Hello World!')
        if self.type is None:
            raise TypingError('Undecided type {}'.format(self))
        return self.type

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 1 if self.type is not None else 0

class ConstraintNetwork(object):
    """
    TODO: It is possible to optimize constraint propagation to consider only
          dirty type variables.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.constraints = []

    def append(self, constraint):
        if False:
            return 10
        self.constraints.append(constraint)

    def propagate(self, typeinfer):
        if False:
            i = 10
            return i + 15
        '\n        Execute all constraints.  Errors are caught and returned as a list.\n        This allows progressing even though some constraints may fail\n        due to lack of information\n        (e.g. imprecise types such as List(undefined)).\n        '
        errors = []
        for constraint in self.constraints:
            loc = constraint.loc
            with typeinfer.warnings.catch_warnings(filename=loc.filename, lineno=loc.line):
                try:
                    constraint(typeinfer)
                except ForceLiteralArg as e:
                    errors.append(e)
                except TypingError as e:
                    _logger.debug('captured error', exc_info=e)
                    new_exc = TypingError(str(e), loc=constraint.loc, highlighting=False)
                    errors.append(utils.chain_exception(new_exc, e))
                except Exception as e:
                    if utils.use_old_style_errors():
                        _logger.debug('captured error', exc_info=e)
                        msg = 'Internal error at {con}.\n{err}\nEnable logging at debug level for details.'
                        new_exc = TypingError(msg.format(con=constraint, err=str(e)), loc=constraint.loc, highlighting=False)
                        errors.append(utils.chain_exception(new_exc, e))
                    elif utils.use_new_style_errors():
                        raise e
                    else:
                        msg = f"Unknown CAPTURED_ERRORS style: '{config.CAPTURED_ERRORS}'."
                        assert 0, msg
        return errors

class Propagate(object):
    """
    A simple constraint for direct propagation of types for assignments.
    """

    def __init__(self, dst, src, loc):
        if False:
            for i in range(10):
                print('nop')
        self.dst = dst
        self.src = src
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            i = 10
            return i + 15
        with new_error_context('typing of assignment at {0}', self.loc, loc=self.loc):
            typeinfer.copy_type(self.src, self.dst, loc=self.loc)
            typeinfer.refine_map[self.dst] = self

    def refine(self, typeinfer, target_type):
        if False:
            return 10
        assert target_type.is_precise()
        typeinfer.add_type(self.src, target_type, unless_locked=True, loc=self.loc)

class ArgConstraint(object):

    def __init__(self, dst, src, loc):
        if False:
            i = 10
            return i + 15
        self.dst = dst
        self.src = src
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            while True:
                i = 10
        with new_error_context('typing of argument at {0}', self.loc):
            typevars = typeinfer.typevars
            src = typevars[self.src]
            if not src.defined:
                return
            ty = src.getone()
            if isinstance(ty, types.Omitted):
                ty = typeinfer.context.resolve_value_type_prefer_literal(ty.value)
            if not ty.is_precise():
                raise TypingError('non-precise type {}'.format(ty))
            typeinfer.add_type(self.dst, ty, loc=self.loc)

class BuildTupleConstraint(object):

    def __init__(self, target, items, loc):
        if False:
            return 10
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            print('Hello World!')
        with new_error_context('typing of tuple at {0}', self.loc):
            typevars = typeinfer.typevars
            tsets = [typevars[i.name].get() for i in self.items]
            for vals in itertools.product(*tsets):
                if vals and all((vals[0] == v for v in vals)):
                    tup = types.UniTuple(dtype=vals[0], count=len(vals))
                else:
                    tup = types.Tuple(vals)
                assert tup.is_precise()
                typeinfer.add_type(self.target, tup, loc=self.loc)

class _BuildContainerConstraint(object):

    def __init__(self, target, items, loc):
        if False:
            i = 10
            return i + 15
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            return 10
        with new_error_context('typing of {0} at {1}', self.container_type, self.loc):
            typevars = typeinfer.typevars
            tsets = [typevars[i.name].get() for i in self.items]
            if not tsets:
                typeinfer.add_type(self.target, self.container_type(types.undefined), loc=self.loc)
            else:
                for typs in itertools.product(*tsets):
                    unified = typeinfer.context.unify_types(*typs)
                    if unified is not None:
                        typeinfer.add_type(self.target, self.container_type(unified), loc=self.loc)

class BuildListConstraint(_BuildContainerConstraint):

    def __init__(self, target, items, loc):
        if False:
            for i in range(10):
                print('nop')
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            for i in range(10):
                print('nop')
        with new_error_context('typing of {0} at {1}', types.List, self.loc):
            typevars = typeinfer.typevars
            tsets = [typevars[i.name].get() for i in self.items]
            if not tsets:
                typeinfer.add_type(self.target, types.List(types.undefined), loc=self.loc)
            else:
                for typs in itertools.product(*tsets):
                    unified = typeinfer.context.unify_types(*typs)
                    if unified is not None:
                        islit = [isinstance(x, types.Literal) for x in typs]
                        iv = None
                        if all(islit):
                            iv = [x.literal_value for x in typs]
                        typeinfer.add_type(self.target, types.List(unified, initial_value=iv), loc=self.loc)
                    else:
                        typeinfer.add_type(self.target, types.LiteralList(typs), loc=self.loc)

class BuildSetConstraint(_BuildContainerConstraint):
    container_type = types.Set

class BuildMapConstraint(object):

    def __init__(self, target, items, special_value, value_indexes, loc):
        if False:
            i = 10
            return i + 15
        self.target = target
        self.items = items
        self.special_value = special_value
        self.value_indexes = value_indexes
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            while True:
                i = 10
        with new_error_context('typing of dict at {0}', self.loc):
            typevars = typeinfer.typevars
            tsets = [(typevars[k.name].getone(), typevars[v.name].getone()) for (k, v) in self.items]
            if not tsets:
                typeinfer.add_type(self.target, types.DictType(types.undefined, types.undefined, self.special_value), loc=self.loc)
            else:
                ktys = [x[0] for x in tsets]
                vtys = [x[1] for x in tsets]
                strkey = all([isinstance(x, types.StringLiteral) for x in ktys])
                literalvty = all([isinstance(x, types.Literal) for x in vtys])
                vt0 = types.unliteral(vtys[0])

                def check(other):
                    if False:
                        print('Hello World!')
                    conv = typeinfer.context.can_convert(other, vt0)
                    return conv is not None and conv < Conversion.unsafe
                homogeneous = all([check(types.unliteral(x)) for x in vtys])
                if len(vtys) == 1:
                    valty = vtys[0]
                    if isinstance(valty, (types.LiteralStrKeyDict, types.List, types.LiteralList)):
                        homogeneous = False
                if strkey and (not homogeneous):
                    resolved_dict = {x: y for (x, y) in zip(ktys, vtys)}
                    ty = types.LiteralStrKeyDict(resolved_dict, self.value_indexes)
                    typeinfer.add_type(self.target, ty, loc=self.loc)
                else:
                    init_value = self.special_value if literalvty else None
                    (key_type, value_type) = tsets[0]
                    typeinfer.add_type(self.target, types.DictType(key_type, value_type, init_value), loc=self.loc)

class ExhaustIterConstraint(object):

    def __init__(self, target, count, iterator, loc):
        if False:
            while True:
                i = 10
        self.target = target
        self.count = count
        self.iterator = iterator
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            while True:
                i = 10
        with new_error_context('typing of exhaust iter at {0}', self.loc):
            typevars = typeinfer.typevars
            for tp in typevars[self.iterator.name].get():
                tp = tp.type if isinstance(tp, types.Optional) else tp
                if isinstance(tp, types.BaseTuple):
                    if len(tp) == self.count:
                        assert tp.is_precise()
                        typeinfer.add_type(self.target, tp, loc=self.loc)
                        break
                    else:
                        msg = (f'wrong tuple length for {self.iterator.name}: ', f'expected {self.count}, got {len(tp)}')
                        raise NumbaValueError(msg)
                elif isinstance(tp, types.IterableType):
                    tup = types.UniTuple(dtype=tp.iterator_type.yield_type, count=self.count)
                    assert tup.is_precise()
                    typeinfer.add_type(self.target, tup, loc=self.loc)
                    break
                else:
                    raise TypingError('failed to unpack {}'.format(tp), loc=self.loc)

class PairFirstConstraint(object):

    def __init__(self, target, pair, loc):
        if False:
            return 10
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            return 10
        with new_error_context('typing of pair-first at {0}', self.loc):
            typevars = typeinfer.typevars
            for tp in typevars[self.pair.name].get():
                if not isinstance(tp, types.Pair):
                    continue
                assert isinstance(tp.first_type, types.UndefinedFunctionType) or tp.first_type.is_precise()
                typeinfer.add_type(self.target, tp.first_type, loc=self.loc)

class PairSecondConstraint(object):

    def __init__(self, target, pair, loc):
        if False:
            return 10
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            i = 10
            return i + 15
        with new_error_context('typing of pair-second at {0}', self.loc):
            typevars = typeinfer.typevars
            for tp in typevars[self.pair.name].get():
                if not isinstance(tp, types.Pair):
                    continue
                assert tp.second_type.is_precise()
                typeinfer.add_type(self.target, tp.second_type, loc=self.loc)

class StaticGetItemConstraint(object):

    def __init__(self, target, value, index, index_var, loc):
        if False:
            for i in range(10):
                print('nop')
        self.target = target
        self.value = value
        self.index = index
        if index_var is not None:
            self.fallback = IntrinsicCallConstraint(target, operator.getitem, (value, index_var), {}, None, loc)
        else:
            self.fallback = None
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            while True:
                i = 10
        with new_error_context('typing of static-get-item at {0}', self.loc):
            typevars = typeinfer.typevars
            for ty in typevars[self.value.name].get():
                sig = typeinfer.context.resolve_static_getitem(value=ty, index=self.index)
                if sig is not None:
                    itemty = sig.return_type
                    typeinfer.add_type(self.target, itemty, loc=self.loc)
                elif self.fallback is not None:
                    self.fallback(typeinfer)

    def get_call_signature(self):
        if False:
            i = 10
            return i + 15
        return self.fallback and self.fallback.get_call_signature()

class TypedGetItemConstraint(object):

    def __init__(self, target, value, dtype, index, loc):
        if False:
            i = 10
            return i + 15
        self.target = target
        self.value = value
        self.dtype = dtype
        self.index = index
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            i = 10
            return i + 15
        with new_error_context('typing of typed-get-item at {0}', self.loc):
            typevars = typeinfer.typevars
            idx_ty = typevars[self.index.name].get()
            ty = typevars[self.value.name].get()
            self.signature = Signature(self.dtype, ty + idx_ty, None)
            typeinfer.add_type(self.target, self.dtype, loc=self.loc)

    def get_call_signature(self):
        if False:
            i = 10
            return i + 15
        return self.signature

def fold_arg_vars(typevars, args, vararg, kws):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fold and resolve the argument variables of a function call.\n    '
    n_pos_args = len(args)
    kwds = [kw for (kw, var) in kws]
    argtypes = [typevars[a.name] for a in args]
    argtypes += [typevars[var.name] for (kw, var) in kws]
    if vararg is not None:
        argtypes.append(typevars[vararg.name])
    if not all((a.defined for a in argtypes)):
        return
    args = tuple((a.getone() for a in argtypes))
    pos_args = args[:n_pos_args]
    if vararg is not None:
        errmsg = '*args in function call should be a tuple, got %s'
        if isinstance(args[-1], types.Literal):
            const_val = args[-1].literal_value
            if not isinstance(const_val, tuple):
                raise TypeError(errmsg % (args[-1],))
            pos_args += const_val
        elif not isinstance(args[-1], types.BaseTuple):
            raise TypeError(errmsg % (args[-1],))
        else:
            pos_args += args[-1].types
        args = args[:-1]
    kw_args = dict(zip(kwds, args[n_pos_args:]))
    return (pos_args, kw_args)

def _is_array_not_precise(arrty):
    if False:
        while True:
            i = 10
    'Check type is array and it is not precise\n    '
    return isinstance(arrty, types.Array) and (not arrty.is_precise())

class CallConstraint(object):
    """Constraint for calling functions.
    Perform case analysis foreach combinations of argument types.
    """
    signature = None

    def __init__(self, target, func, args, kws, vararg, loc):
        if False:
            for i in range(10):
                print('nop')
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws or {}
        self.vararg = vararg
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            i = 10
            return i + 15
        msg = 'typing of call at {0}\n'.format(self.loc)
        with new_error_context(msg):
            typevars = typeinfer.typevars
            with new_error_context('resolving caller type: {}'.format(self.func)):
                fnty = typevars[self.func].getone()
            with new_error_context('resolving callee type: {0}', fnty):
                self.resolve(typeinfer, typevars, fnty)

    def resolve(self, typeinfer, typevars, fnty):
        if False:
            for i in range(10):
                print('nop')
        assert fnty
        context = typeinfer.context
        r = fold_arg_vars(typevars, self.args, self.vararg, self.kws)
        if r is None:
            return
        (pos_args, kw_args) = r
        for a in itertools.chain(pos_args, kw_args.values()):
            if not a.is_precise() and (not isinstance(a, types.Array)):
                return
        if isinstance(fnty, types.TypeRef):
            fnty = fnty.instance_type
        try:
            sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
        except ForceLiteralArg as e:
            folding_args = (fnty.this,) + tuple(self.args) if isinstance(fnty, types.BoundFunction) else self.args
            folded = e.fold_arguments(folding_args, self.kws)
            requested = set()
            unsatisfied = set()
            for idx in e.requested_args:
                maybe_arg = typeinfer.func_ir.get_definition(folded[idx])
                if isinstance(maybe_arg, ir.Arg):
                    requested.add(maybe_arg.index)
                else:
                    unsatisfied.add(idx)
            if unsatisfied:
                raise TypingError('Cannot request literal type.', loc=self.loc)
            elif requested:
                raise ForceLiteralArg(requested, loc=self.loc)
        if sig is None:
            headtemp = 'Invalid use of {0} with parameters ({1})'
            args = [str(a) for a in pos_args]
            args += ['%s=%s' % (k, v) for (k, v) in sorted(kw_args.items())]
            head = headtemp.format(fnty, ', '.join(map(str, args)))
            desc = context.explain_function_type(fnty)
            msg = '\n'.join([head, desc])
            raise TypingError(msg)
        typeinfer.add_type(self.target, sig.return_type, loc=self.loc)
        if isinstance(fnty, types.BoundFunction) and sig.recvr is not None and (sig.recvr != fnty.this):
            refined_this = context.unify_pairs(sig.recvr, fnty.this)
            if refined_this is None and fnty.this.is_precise() and sig.recvr.is_precise():
                msg = 'Cannot refine type {} to {}'.format(sig.recvr, fnty.this)
                raise TypingError(msg, loc=self.loc)
            if refined_this is not None and refined_this.is_precise():
                refined_fnty = fnty.copy(this=refined_this)
                typeinfer.propagate_refined_type(self.func, refined_fnty)
        if not sig.return_type.is_precise():
            target = typevars[self.target]
            if target.defined:
                targetty = target.getone()
                if context.unify_pairs(targetty, sig.return_type) == targetty:
                    sig = sig.replace(return_type=targetty)
        self.signature = sig
        self._add_refine_map(typeinfer, typevars, sig)

    def _add_refine_map(self, typeinfer, typevars, sig):
        if False:
            i = 10
            return i + 15
        'Add this expression to the refine_map base on the type of target_type\n        '
        target_type = typevars[self.target].getone()
        if isinstance(target_type, types.Array) and isinstance(sig.return_type.dtype, types.Undefined):
            typeinfer.refine_map[self.target] = self
        if isinstance(target_type, types.DictType) and (not target_type.is_precise()):
            typeinfer.refine_map[self.target] = self

    def refine(self, typeinfer, updated_type):
        if False:
            i = 10
            return i + 15
        if self.func == operator.getitem:
            aryty = typeinfer.typevars[self.args[0].name].getone()
            if _is_array_not_precise(aryty):
                assert updated_type.is_precise()
                newtype = aryty.copy(dtype=updated_type.dtype)
                typeinfer.add_type(self.args[0].name, newtype, loc=self.loc)
        else:
            m = 'no type refinement implemented for function {} updating to {}'
            raise TypingError(m.format(self.func, updated_type))

    def get_call_signature(self):
        if False:
            i = 10
            return i + 15
        return self.signature

class IntrinsicCallConstraint(CallConstraint):

    def __call__(self, typeinfer):
        if False:
            return 10
        with new_error_context('typing of intrinsic-call at {0}', self.loc):
            fnty = self.func
            if fnty in utils.OPERATORS_TO_BUILTINS:
                fnty = typeinfer.resolve_value_type(None, fnty)
            self.resolve(typeinfer, typeinfer.typevars, fnty=fnty)

class GetAttrConstraint(object):

    def __init__(self, target, attr, value, loc, inst):
        if False:
            i = 10
            return i + 15
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc
        self.inst = inst

    def __call__(self, typeinfer):
        if False:
            return 10
        with new_error_context('typing of get attribute at {0}', self.loc):
            typevars = typeinfer.typevars
            valtys = typevars[self.value.name].get()
            for ty in valtys:
                attrty = typeinfer.context.resolve_getattr(ty, self.attr)
                if attrty is None:
                    raise UntypedAttributeError(ty, self.attr, loc=self.inst.loc)
                else:
                    assert attrty.is_precise()
                    typeinfer.add_type(self.target, attrty, loc=self.loc)
            typeinfer.refine_map[self.target] = self

    def refine(self, typeinfer, target_type):
        if False:
            while True:
                i = 10
        if isinstance(target_type, types.BoundFunction):
            recvr = target_type.this
            assert recvr.is_precise()
            typeinfer.add_type(self.value.name, recvr, loc=self.loc)
            source_constraint = typeinfer.refine_map.get(self.value.name)
            if source_constraint is not None:
                source_constraint.refine(typeinfer, recvr)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'resolving type of attribute "{attr}" of "{value}"'.format(value=self.value, attr=self.attr)

class SetItemRefinement(object):
    """A mixin class to provide the common refinement logic in setitem
    and static setitem.
    """

    def _refine_target_type(self, typeinfer, targetty, idxty, valty, sig):
        if False:
            for i in range(10):
                print('nop')
        'Refine the target-type given the known index type and value type.\n        '
        if _is_array_not_precise(targetty):
            typeinfer.add_type(self.target.name, sig.args[0], loc=self.loc)
        if isinstance(targetty, types.DictType):
            if not targetty.is_precise():
                refined = targetty.refine(idxty, valty)
                typeinfer.add_type(self.target.name, refined, loc=self.loc)
            elif isinstance(targetty, types.LiteralStrKeyDict):
                typeinfer.add_type(self.target.name, types.DictType(idxty, valty), loc=self.loc)

class SetItemConstraint(SetItemRefinement):

    def __init__(self, target, index, value, loc):
        if False:
            i = 10
            return i + 15
        self.target = target
        self.index = index
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            for i in range(10):
                print('nop')
        with new_error_context('typing of setitem at {0}', self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.index, self.value))):
                return
            targetty = typevars[self.target.name].getone()
            idxty = typevars[self.index.name].getone()
            valty = typevars[self.value.name].getone()
            sig = typeinfer.context.resolve_setitem(targetty, idxty, valty)
            if sig is None:
                raise TypingError('Cannot resolve setitem: %s[%s] = %s' % (targetty, idxty, valty), loc=self.loc)
            self.signature = sig
            self._refine_target_type(typeinfer, targetty, idxty, valty, sig)

    def get_call_signature(self):
        if False:
            i = 10
            return i + 15
        return self.signature

class StaticSetItemConstraint(SetItemRefinement):

    def __init__(self, target, index, index_var, value, loc):
        if False:
            while True:
                i = 10
        self.target = target
        self.index = index
        self.index_var = index_var
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            while True:
                i = 10
        with new_error_context('typing of staticsetitem at {0}', self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.index_var, self.value))):
                return
            targetty = typevars[self.target.name].getone()
            idxty = typevars[self.index_var.name].getone()
            valty = typevars[self.value.name].getone()
            sig = typeinfer.context.resolve_static_setitem(targetty, self.index, valty)
            if sig is None:
                sig = typeinfer.context.resolve_setitem(targetty, idxty, valty)
            if sig is None:
                raise TypingError('Cannot resolve setitem: %s[%r] = %s' % (targetty, self.index, valty), loc=self.loc)
            self.signature = sig
            self._refine_target_type(typeinfer, targetty, idxty, valty, sig)

    def get_call_signature(self):
        if False:
            return 10
        return self.signature

class DelItemConstraint(object):

    def __init__(self, target, index, loc):
        if False:
            return 10
        self.target = target
        self.index = index
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            for i in range(10):
                print('nop')
        with new_error_context('typing of delitem at {0}', self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.index))):
                return
            targetty = typevars[self.target.name].getone()
            idxty = typevars[self.index.name].getone()
            sig = typeinfer.context.resolve_delitem(targetty, idxty)
            if sig is None:
                raise TypingError('Cannot resolve delitem: %s[%s]' % (targetty, idxty), loc=self.loc)
            self.signature = sig

    def get_call_signature(self):
        if False:
            for i in range(10):
                print('nop')
        return self.signature

class SetAttrConstraint(object):

    def __init__(self, target, attr, value, loc):
        if False:
            i = 10
            return i + 15
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            return 10
        with new_error_context('typing of set attribute {0!r} at {1}', self.attr, self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.value))):
                return
            targetty = typevars[self.target.name].getone()
            valty = typevars[self.value.name].getone()
            sig = typeinfer.context.resolve_setattr(targetty, self.attr, valty)
            if sig is None:
                raise TypingError('Cannot resolve setattr: (%s).%s = %s' % (targetty, self.attr, valty), loc=self.loc)
            self.signature = sig

    def get_call_signature(self):
        if False:
            i = 10
            return i + 15
        return self.signature

class PrintConstraint(object):

    def __init__(self, args, vararg, loc):
        if False:
            return 10
        self.args = args
        self.vararg = vararg
        self.loc = loc

    def __call__(self, typeinfer):
        if False:
            i = 10
            return i + 15
        typevars = typeinfer.typevars
        r = fold_arg_vars(typevars, self.args, self.vararg, {})
        if r is None:
            return
        (pos_args, kw_args) = r
        fnty = typeinfer.context.resolve_value_type(print)
        assert fnty is not None
        sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
        self.signature = sig

    def get_call_signature(self):
        if False:
            print('Hello World!')
        return self.signature

class TypeVarMap(dict):

    def set_context(self, context):
        if False:
            print('Hello World!')
        self.context = context

    def __getitem__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name not in self:
            self[name] = TypeVar(self.context, name)
        return super(TypeVarMap, self).__getitem__(name)

    def __setitem__(self, name, value):
        if False:
            i = 10
            return i + 15
        assert isinstance(name, str)
        if name in self:
            raise KeyError('Cannot redefine typevar %s' % name)
        else:
            super(TypeVarMap, self).__setitem__(name, value)
_temporary_dispatcher_map = {}
_temporary_dispatcher_map_ref_count = defaultdict(int)

@contextlib.contextmanager
def register_dispatcher(disp):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register a Dispatcher for inference while it is not yet stored\n    as global or closure variable (e.g. during execution of the @jit()\n    call).  This allows resolution of recursive calls with eager\n    compilation.\n    '
    assert callable(disp)
    assert callable(disp.py_func)
    name = disp.py_func.__name__
    _temporary_dispatcher_map[name] = disp
    _temporary_dispatcher_map_ref_count[name] += 1
    try:
        yield
    finally:
        _temporary_dispatcher_map_ref_count[name] -= 1
        if not _temporary_dispatcher_map_ref_count[name]:
            del _temporary_dispatcher_map[name]
typeinfer_extensions = {}

class TypeInferer(object):
    """
    Operates on block that shares the same ir.Scope.
    """

    def __init__(self, context, func_ir, warnings):
        if False:
            i = 10
            return i + 15
        self.context = context
        self.blocks = OrderedDict()
        for k in sorted(func_ir.blocks.keys()):
            self.blocks[k] = func_ir.blocks[k]
        self.generator_info = func_ir.generator_info
        self.func_id = func_ir.func_id
        self.func_ir = func_ir
        self.typevars = TypeVarMap()
        self.typevars.set_context(context)
        self.constraints = ConstraintNetwork()
        self.warnings = warnings
        self.arg_names = {}
        self.assumed_immutables = set()
        self.calls = []
        self.calltypes = utils.UniqueDict()
        self.refine_map = {}
        if config.DEBUG or config.DEBUG_TYPEINFER:
            self.debug = TypeInferDebug(self)
        else:
            self.debug = NullDebug()
        self._skip_recursion = False

    def copy(self, skip_recursion=False):
        if False:
            while True:
                i = 10
        clone = TypeInferer(self.context, self.func_ir, self.warnings)
        clone.arg_names = self.arg_names.copy()
        clone._skip_recursion = skip_recursion
        for (k, v) in self.typevars.items():
            if not v.locked and v.defined:
                clone.typevars[k].add_type(v.getone(), loc=v.define_loc)
        return clone

    def _mangle_arg_name(self, name):
        if False:
            while True:
                i = 10
        return 'arg.%s' % (name,)

    def _get_return_vars(self):
        if False:
            return 10
        rets = []
        for blk in self.blocks.values():
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                rets.append(inst.value)
        return rets

    def get_argument_types(self):
        if False:
            while True:
                i = 10
        return [self.typevars[k].getone() for k in self.arg_names.values()]

    def seed_argument(self, name, index, typ):
        if False:
            i = 10
            return i + 15
        name = self._mangle_arg_name(name)
        self.seed_type(name, typ)
        self.arg_names[index] = name

    def seed_type(self, name, typ):
        if False:
            for i in range(10):
                print('nop')
        'All arguments should be seeded.\n        '
        self.lock_type(name, typ, loc=None)

    def seed_return(self, typ):
        if False:
            return 10
        'Seeding of return value is optional.\n        '
        for var in self._get_return_vars():
            self.lock_type(var.name, typ, loc=None)

    def build_constraint(self):
        if False:
            return 10
        for blk in self.blocks.values():
            for inst in blk.body:
                self.constrain_statement(inst)

    def return_types_from_partial(self):
        if False:
            print('Hello World!')
        '\n        Resume type inference partially to deduce the return type.\n        Note: No side-effect to `self`.\n\n        Returns the inferred return type or None if it cannot deduce the return\n        type.\n        '
        cloned = self.copy(skip_recursion=True)
        cloned.build_constraint()
        cloned.propagate(raise_errors=False)
        rettypes = set()
        for retvar in cloned._get_return_vars():
            if retvar.name in cloned.typevars:
                typevar = cloned.typevars[retvar.name]
                if typevar and typevar.defined:
                    rettypes.add(types.unliteral(typevar.getone()))
        if not rettypes:
            return
        return cloned._unify_return_types(rettypes)

    def propagate(self, raise_errors=True):
        if False:
            i = 10
            return i + 15
        newtoken = self.get_state_token()
        oldtoken = None
        while newtoken != oldtoken:
            self.debug.propagate_started()
            oldtoken = newtoken
            errors = self.constraints.propagate(self)
            newtoken = self.get_state_token()
            self.debug.propagate_finished()
        if errors:
            if raise_errors:
                force_lit_args = [e for e in errors if isinstance(e, ForceLiteralArg)]
                if not force_lit_args:
                    raise errors[0]
                else:
                    raise reduce(operator.or_, force_lit_args)
            else:
                return errors

    def add_type(self, var, tp, loc, unless_locked=False):
        if False:
            return 10
        assert isinstance(var, str), type(var)
        tv = self.typevars[var]
        if unless_locked and tv.locked:
            return
        oldty = tv.type
        unified = tv.add_type(tp, loc=loc)
        if unified != oldty:
            self.propagate_refined_type(var, unified)

    def add_calltype(self, inst, signature):
        if False:
            return 10
        assert signature is not None
        self.calltypes[inst] = signature

    def copy_type(self, src_var, dest_var, loc):
        if False:
            while True:
                i = 10
        self.typevars[dest_var].union(self.typevars[src_var], loc=loc)

    def lock_type(self, var, tp, loc, literal_value=NOTSET):
        if False:
            while True:
                i = 10
        tv = self.typevars[var]
        tv.lock(tp, loc=loc, literal_value=literal_value)

    def propagate_refined_type(self, updated_var, updated_type):
        if False:
            for i in range(10):
                print('nop')
        source_constraint = self.refine_map.get(updated_var)
        if source_constraint is not None:
            source_constraint.refine(self, updated_type)

    def unify(self, raise_errors=True):
        if False:
            print('Hello World!')
        '\n        Run the final unification pass over all inferred types, and\n        catch imprecise types.\n        '
        typdict = utils.UniqueDict()

        def find_offender(name, exhaustive=False):
            if False:
                i = 10
                return i + 15
            offender = None
            for block in self.func_ir.blocks.values():
                offender = block.find_variable_assignment(name)
                if offender is not None:
                    if not exhaustive:
                        break
                    try:
                        hasattr(offender.value, 'name')
                        offender_value = offender.value.name
                    except (AttributeError, KeyError):
                        break
                    orig_offender = offender
                    if offender_value.startswith('$'):
                        offender = find_offender(offender_value, exhaustive=exhaustive)
                        if offender is None:
                            offender = orig_offender
                    break
            return offender

        def diagnose_imprecision(offender):
            if False:
                i = 10
                return i + 15
            list_msg = '\n\nFor Numba to be able to compile a list, the list must have a known and\nprecise type that can be inferred from the other variables. Whilst sometimes\nthe type of empty lists can be inferred, this is not always the case, see this\ndocumentation for help:\n\nhttps://numba.readthedocs.io/en/stable/user/troubleshoot.html#my-code-has-an-untyped-list-problem\n'
            if offender is not None:
                if hasattr(offender, 'value'):
                    if hasattr(offender.value, 'op'):
                        if offender.value.op == 'build_list':
                            return list_msg
                        elif offender.value.op == 'call':
                            try:
                                call_name = offender.value.func.name
                                offender = find_offender(call_name)
                                if isinstance(offender.value, ir.Global):
                                    if offender.value.name == 'list':
                                        return list_msg
                            except (AttributeError, KeyError):
                                pass
            return ''

        def check_var(name):
            if False:
                return 10
            tv = self.typevars[name]
            if not tv.defined:
                if raise_errors:
                    offender = find_offender(name)
                    val = getattr(offender, 'value', 'unknown operation')
                    loc = getattr(offender, 'loc', ir.unknown_loc)
                    msg = "Type of variable '%s' cannot be determined, operation: %s, location: %s"
                    raise TypingError(msg % (var, val, loc), loc)
                else:
                    typdict[var] = types.unknown
                    return
            tp = tv.getone()
            if isinstance(tp, types.UndefinedFunctionType):
                tp = tp.get_precise()
            if not tp.is_precise():
                offender = find_offender(name, exhaustive=True)
                msg = "Cannot infer the type of variable '%s'%s, have imprecise type: %s. %s"
                istmp = ' (temporary variable)' if var.startswith('$') else ''
                loc = getattr(offender, 'loc', ir.unknown_loc)
                extra_msg = diagnose_imprecision(offender)
                if raise_errors:
                    raise TypingError(msg % (var, istmp, tp, extra_msg), loc)
                else:
                    typdict[var] = types.unknown
                    return
            else:
                typdict[var] = tp
        temps = set((k for k in self.typevars if not k[0].isalpha()))
        others = set(self.typevars) - temps
        for var in sorted(others):
            check_var(var)
        for var in sorted(temps):
            check_var(var)
        try:
            retty = self.get_return_type(typdict)
        except Exception as e:
            if raise_errors:
                raise e
            else:
                retty = None
        else:
            typdict = utils.UniqueDict(typdict, **{v.name: retty for v in self._get_return_vars()})
        try:
            fntys = self.get_function_types(typdict)
        except Exception as e:
            if raise_errors:
                raise e
            else:
                fntys = None
        if self.generator_info:
            retty = self.get_generator_type(typdict, retty, raise_errors=raise_errors)
        self.debug.unify_finished(typdict, retty, fntys)
        return (typdict, retty, fntys)

    def get_generator_type(self, typdict, retty, raise_errors=True):
        if False:
            while True:
                i = 10
        gi = self.generator_info
        arg_types = [None] * len(self.arg_names)
        for (index, name) in self.arg_names.items():
            arg_types[index] = typdict[name]
        state_types = None
        try:
            state_types = [typdict[var_name] for var_name in gi.state_vars]
        except KeyError:
            msg = 'Cannot type generator: state variable types cannot be found'
            if raise_errors:
                raise TypingError(msg)
            state_types = [types.unknown for _ in gi.state_vars]
        yield_types = None
        try:
            yield_types = [typdict[y.inst.value.name] for y in gi.get_yield_points()]
        except KeyError:
            msg = 'Cannot type generator: yield type cannot be found'
            if raise_errors:
                raise TypingError(msg)
        if not yield_types:
            msg = 'Cannot type generator: it does not yield any value'
            if raise_errors:
                raise TypingError(msg)
            yield_types = [types.unknown for _ in gi.get_yield_points()]
        if not yield_types or all(yield_types) == types.unknown:
            return types.Generator(self.func_id.func, types.unknown, arg_types, state_types, has_finalizer=True)
        yield_type = self.context.unify_types(*yield_types)
        if yield_type is None or isinstance(yield_type, types.Optional):
            msg = 'Cannot type generator: cannot unify yielded types %s'
            yp_highlights = []
            for y in gi.get_yield_points():
                msg = _termcolor.errmsg("Yield of: IR '%s', type '%s', location: %s")
                yp_highlights.append(msg % (str(y.inst), typdict[y.inst.value.name], y.inst.loc.strformat()))
            explain_ty = set()
            for ty in yield_types:
                if isinstance(ty, types.Optional):
                    explain_ty.add(ty.type)
                    explain_ty.add(types.NoneType('none'))
                else:
                    explain_ty.add(ty)
            if raise_errors:
                raise TypingError("Can't unify yield type from the following types: %s" % ', '.join(sorted(map(str, explain_ty))) + '\n\n' + '\n'.join(yp_highlights))
        return types.Generator(self.func_id.func, yield_type, arg_types, state_types, has_finalizer=True)

    def get_function_types(self, typemap):
        if False:
            print('Hello World!')
        '\n        Fill and return the calltypes map.\n        '
        calltypes = self.calltypes
        for (call, constraint) in self.calls:
            calltypes[call] = constraint.get_call_signature()
        return calltypes

    def _unify_return_types(self, rettypes):
        if False:
            while True:
                i = 10
        if rettypes:
            unified = self.context.unify_types(*rettypes)
            if isinstance(unified, types.FunctionType):
                return unified
            if unified is None or not unified.is_precise():

                def check_type(atype):
                    if False:
                        for i in range(10):
                            print('nop')
                    lst = []
                    for (k, v) in self.typevars.items():
                        if atype == v.type:
                            lst.append(k)
                    returns = {}
                    for x in reversed(lst):
                        for block in self.func_ir.blocks.values():
                            for instr in block.find_insts(ir.Return):
                                value = instr.value
                                if isinstance(value, ir.Var):
                                    name = value.name
                                else:
                                    pass
                                if x == name:
                                    returns[x] = instr
                                    break
                    interped = ''
                    for (name, offender) in returns.items():
                        loc = getattr(offender, 'loc', ir.unknown_loc)
                        msg = "Return of: IR name '%s', type '%s', location: %s"
                        interped = msg % (name, atype, loc.strformat())
                    return interped
                problem_str = []
                for xtype in rettypes:
                    problem_str.append(_termcolor.errmsg(check_type(xtype)))
                raise TypingError("Can't unify return type from the following types: %s" % ', '.join(sorted(map(str, rettypes))) + '\n' + '\n'.join(problem_str))
            return unified
        else:
            return types.none

    def get_return_type(self, typemap):
        if False:
            return 10
        rettypes = set()
        for var in self._get_return_vars():
            rettypes.add(typemap[var.name])
        return self._unify_return_types(rettypes)

    def get_state_token(self):
        if False:
            while True:
                i = 10
        'The algorithm is monotonic.  It can only grow or "refine" the\n        typevar map.\n        '
        return [tv.type for (name, tv) in sorted(self.typevars.items())]

    def constrain_statement(self, inst):
        if False:
            return 10
        if isinstance(inst, ir.Assign):
            self.typeof_assign(inst)
        elif isinstance(inst, ir.SetItem):
            self.typeof_setitem(inst)
        elif isinstance(inst, ir.StaticSetItem):
            self.typeof_static_setitem(inst)
        elif isinstance(inst, ir.DelItem):
            self.typeof_delitem(inst)
        elif isinstance(inst, ir.SetAttr):
            self.typeof_setattr(inst)
        elif isinstance(inst, ir.Print):
            self.typeof_print(inst)
        elif isinstance(inst, ir.StoreMap):
            self.typeof_storemap(inst)
        elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return, ir.Del)):
            pass
        elif isinstance(inst, (ir.DynamicRaise, ir.DynamicTryRaise)):
            pass
        elif isinstance(inst, (ir.StaticRaise, ir.StaticTryRaise)):
            pass
        elif isinstance(inst, ir.PopBlock):
            pass
        elif type(inst) in typeinfer_extensions:
            f = typeinfer_extensions[type(inst)]
            f(inst, self)
        else:
            msg = 'Unsupported constraint encountered: %s' % inst
            raise UnsupportedError(msg, loc=inst.loc)

    def typeof_setitem(self, inst):
        if False:
            i = 10
            return i + 15
        constraint = SetItemConstraint(target=inst.target, index=inst.index, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_storemap(self, inst):
        if False:
            print('Hello World!')
        constraint = SetItemConstraint(target=inst.dct, index=inst.key, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_static_setitem(self, inst):
        if False:
            print('Hello World!')
        constraint = StaticSetItemConstraint(target=inst.target, index=inst.index, index_var=inst.index_var, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_delitem(self, inst):
        if False:
            for i in range(10):
                print('nop')
        constraint = DelItemConstraint(target=inst.target, index=inst.index, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_setattr(self, inst):
        if False:
            return 10
        constraint = SetAttrConstraint(target=inst.target, attr=inst.attr, value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_print(self, inst):
        if False:
            i = 10
            return i + 15
        constraint = PrintConstraint(args=inst.args, vararg=inst.vararg, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst, constraint))

    def typeof_assign(self, inst):
        if False:
            for i in range(10):
                print('nop')
        value = inst.value
        if isinstance(value, ir.Const):
            self.typeof_const(inst, inst.target, value.value)
        elif isinstance(value, ir.Var):
            self.constraints.append(Propagate(dst=inst.target.name, src=value.name, loc=inst.loc))
        elif isinstance(value, (ir.Global, ir.FreeVar)):
            self.typeof_global(inst, inst.target, value)
        elif isinstance(value, ir.Arg):
            self.typeof_arg(inst, inst.target, value)
        elif isinstance(value, ir.Expr):
            self.typeof_expr(inst, inst.target, value)
        elif isinstance(value, ir.Yield):
            self.typeof_yield(inst, inst.target, value)
        else:
            msg = 'Unsupported assignment encountered: %s %s' % (type(value), str(value))
            raise UnsupportedError(msg, loc=inst.loc)

    def resolve_value_type(self, inst, val):
        if False:
            return 10
        '\n        Resolve the type of a simple Python value, such as can be\n        represented by literals.\n        '
        try:
            return self.context.resolve_value_type(val)
        except ValueError as e:
            msg = str(e)
        raise TypingError(msg, loc=inst.loc)

    def typeof_arg(self, inst, target, arg):
        if False:
            i = 10
            return i + 15
        src_name = self._mangle_arg_name(arg.name)
        self.constraints.append(ArgConstraint(dst=target.name, src=src_name, loc=inst.loc))

    def typeof_const(self, inst, target, const):
        if False:
            while True:
                i = 10
        ty = self.resolve_value_type(inst, const)
        if inst.value.use_literal_type:
            lit = types.maybe_literal(value=const)
        else:
            lit = None
        self.add_type(target.name, lit or ty, loc=inst.loc)

    def typeof_yield(self, inst, target, yield_):
        if False:
            while True:
                i = 10
        self.add_type(target.name, types.none, loc=inst.loc)

    def sentry_modified_builtin(self, inst, gvar):
        if False:
            return 10
        '\n        Ensure that builtins are not modified.\n        '
        if gvar.name == 'range' and gvar.value is not range:
            bad = True
        elif gvar.name == 'slice' and gvar.value is not slice:
            bad = True
        elif gvar.name == 'len' and gvar.value is not len:
            bad = True
        else:
            bad = False
        if bad:
            raise TypingError("Modified builtin '%s'" % gvar.name, loc=inst.loc)

    def resolve_call(self, fnty, pos_args, kw_args):
        if False:
            return 10
        '\n        Resolve a call to a given function type.  A signature is returned.\n        '
        if isinstance(fnty, types.FunctionType):
            return fnty.get_call_type(self, pos_args, kw_args)
        if isinstance(fnty, types.RecursiveCall) and (not self._skip_recursion):
            disp = fnty.dispatcher_type.dispatcher
            (pysig, args) = disp.fold_argument_types(pos_args, kw_args)
            frame = self.context.callstack.match(disp.py_func, args)
            if frame is None:
                sig = self.context.resolve_function_type(fnty.dispatcher_type, pos_args, kw_args)
                fndesc = disp.overloads[args].fndesc
                qual = qualifying_prefix(fndesc.modname, fndesc.qualname)
                fnty.add_overloads(args, qual, fndesc.uid)
                return sig
            fnid = frame.func_id
            qual = qualifying_prefix(fnid.modname, fnid.func_qualname)
            fnty.add_overloads(args, qual, fnid.unique_id)
            return_type = frame.typeinfer.return_types_from_partial()
            if return_type is None:
                raise TypingError('cannot type infer runaway recursion')
            sig = typing.signature(return_type, *args)
            sig = sig.replace(pysig=pysig)
            frame.add_return_type(return_type)
            return sig
        else:
            return self.context.resolve_function_type(fnty, pos_args, kw_args)

    def typeof_global(self, inst, target, gvar):
        if False:
            i = 10
            return i + 15
        try:
            typ = self.resolve_value_type(inst, gvar.value)
        except TypingError as e:
            if gvar.name == self.func_id.func_name and gvar.name in _temporary_dispatcher_map:
                typ = types.Dispatcher(_temporary_dispatcher_map[gvar.name])
            else:
                from numba.misc import special
                nm = gvar.name
                func_glbls = self.func_id.func.__globals__
                if nm not in func_glbls.keys() and nm not in special.__all__ and (nm not in __builtins__.keys()) and (nm not in self.func_id.code.co_freevars):
                    errstr = "NameError: name '%s' is not defined"
                    msg = _termcolor.errmsg(errstr % nm)
                    e.patch_message(msg)
                    raise
                else:
                    msg = _termcolor.errmsg("Untyped global name '%s':" % nm)
                msg += ' %s'
                if nm in special.__all__:
                    tmp = "\n'%s' looks like a Numba internal function, has it been imported (i.e. 'from numba import %s')?\n" % (nm, nm)
                    msg += _termcolor.errmsg(tmp)
                e.patch_message(msg % e)
                raise
        if isinstance(typ, types.Dispatcher) and typ.dispatcher.is_compiling:
            callstack = self.context.callstack
            callframe = callstack.findfirst(typ.dispatcher.py_func)
            if callframe is not None:
                typ = types.RecursiveCall(typ)
            else:
                raise NotImplementedError('call to %s: unsupported recursion' % typ.dispatcher)
        if isinstance(typ, types.Array):
            typ = typ.copy(readonly=True)
        if isinstance(typ, types.BaseAnonymousTuple):
            literaled = [types.maybe_literal(x) for x in gvar.value]
            if all(literaled):
                typ = types.Tuple(literaled)

            def mark_array_ro(tup):
                if False:
                    for i in range(10):
                        print('nop')
                newtup = []
                for item in tup.types:
                    if isinstance(item, types.Array):
                        item = item.copy(readonly=True)
                    elif isinstance(item, types.BaseAnonymousTuple):
                        item = mark_array_ro(item)
                    newtup.append(item)
                return types.BaseTuple.from_types(newtup)
            typ = mark_array_ro(typ)
        self.sentry_modified_builtin(inst, gvar)
        lit = types.maybe_literal(gvar.value)
        tv = self.typevars[target.name]
        if tv.locked:
            tv.add_type(lit or typ, loc=inst.loc)
        else:
            self.lock_type(target.name, lit or typ, loc=inst.loc)
        self.assumed_immutables.add(inst)

    def typeof_expr(self, inst, target, expr):
        if False:
            i = 10
            return i + 15
        if expr.op == 'call':
            self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext'):
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value)
        elif expr.op == 'exhaust_iter':
            constraint = ExhaustIterConstraint(target.name, count=expr.count, iterator=expr.value, loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'pair_first':
            constraint = PairFirstConstraint(target.name, pair=expr.value, loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'pair_second':
            constraint = PairSecondConstraint(target.name, pair=expr.value, loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'inplace_binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'unary':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.value)
        elif expr.op == 'static_getitem':
            constraint = StaticGetItemConstraint(target.name, value=expr.value, index=expr.index, index_var=expr.index_var, loc=expr.loc)
            self.constraints.append(constraint)
            self.calls.append((inst.value, constraint))
        elif expr.op == 'getitem':
            self.typeof_intrinsic_call(inst, target, operator.getitem, expr.value, expr.index)
        elif expr.op == 'typed_getitem':
            constraint = TypedGetItemConstraint(target.name, value=expr.value, dtype=expr.dtype, index=expr.index, loc=expr.loc)
            self.constraints.append(constraint)
            self.calls.append((inst.value, constraint))
        elif expr.op == 'getattr':
            constraint = GetAttrConstraint(target.name, attr=expr.attr, value=expr.value, loc=inst.loc, inst=inst)
            self.constraints.append(constraint)
        elif expr.op == 'build_tuple':
            constraint = BuildTupleConstraint(target.name, items=expr.items, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_list':
            constraint = BuildListConstraint(target.name, items=expr.items, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_set':
            constraint = BuildSetConstraint(target.name, items=expr.items, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_map':
            constraint = BuildMapConstraint(target.name, items=expr.items, special_value=expr.literal_value, value_indexes=expr.value_indexes, loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'cast':
            self.constraints.append(Propagate(dst=target.name, src=expr.value.name, loc=inst.loc))
        elif expr.op == 'phi':
            for iv in expr.incoming_values:
                if iv is not ir.UNDEFINED:
                    self.constraints.append(Propagate(dst=target.name, src=iv.name, loc=inst.loc))
        elif expr.op == 'make_function':
            self.lock_type(target.name, types.MakeFunctionLiteral(expr), loc=inst.loc, literal_value=expr)
        else:
            msg = 'Unsupported op-code encountered: %s' % expr
            raise UnsupportedError(msg, loc=inst.loc)

    def typeof_call(self, inst, target, call):
        if False:
            while True:
                i = 10
        constraint = CallConstraint(target.name, call.func.name, call.args, call.kws, call.vararg, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst.value, constraint))

    def typeof_intrinsic_call(self, inst, target, func, *args):
        if False:
            for i in range(10):
                print('nop')
        constraint = IntrinsicCallConstraint(target.name, func, args, kws=(), vararg=None, loc=inst.loc)
        self.constraints.append(constraint)
        self.calls.append((inst.value, constraint))

class NullDebug(object):

    def propagate_started(self):
        if False:
            return 10
        pass

    def propagate_finished(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def unify_finished(self, typdict, retty, fntys):
        if False:
            while True:
                i = 10
        pass

class TypeInferDebug(object):

    def __init__(self, typeinfer):
        if False:
            return 10
        self.typeinfer = typeinfer

    def _dump_state(self):
        if False:
            for i in range(10):
                print('nop')
        print('---- type variables ----')
        pprint([v for (k, v) in sorted(self.typeinfer.typevars.items())])

    def propagate_started(self):
        if False:
            for i in range(10):
                print('nop')
        print('propagate'.center(80, '-'))

    def propagate_finished(self):
        if False:
            for i in range(10):
                print('nop')
        self._dump_state()

    def unify_finished(self, typdict, retty, fntys):
        if False:
            i = 10
            return i + 15
        print('Variable types'.center(80, '-'))
        pprint(typdict)
        print('Return type'.center(80, '-'))
        pprint(retty)
        print('Call types'.center(80, '-'))
        pprint(fntys)