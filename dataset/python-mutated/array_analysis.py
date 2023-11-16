import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import mk_unique_var, find_topo_order, dprint_func_ir, get_global_func_typ, guard, require, get_definition, find_callname, find_build_sequence, find_const, is_namedtuple_class, build_definitions, find_potential_aliases, get_canonical_alias, GuardException
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
UNKNOWN_CLASS = -1
CONST_CLASS = 0
MAP_TYPES = [numpy.ufunc]
array_analysis_extensions = {}
array_creation = ['empty', 'zeros', 'ones', 'full']
random_int_args = ['rand', 'randn']
random_1arg_size = ['ranf', 'random_sample', 'sample', 'random', 'standard_normal']
random_2arg_sizelast = ['chisquare', 'weibull', 'power', 'geometric', 'exponential', 'poisson', 'rayleigh']
random_3arg_sizelast = ['normal', 'uniform', 'beta', 'binomial', 'f', 'gamma', 'lognormal', 'laplace']
random_calls = random_int_args + random_1arg_size + random_2arg_sizelast + random_3arg_sizelast + ['randint', 'triangular']

@intrinsic
def wrap_index(typingctx, idx, size):
    if False:
        while True:
            i = 10
    '\n    Calculate index value "idx" relative to a size "size" value as\n    (idx % size), where "size" is known to be positive.\n    Note that we use the mod(%) operation here instead of\n    (idx < 0 ? idx + size : idx) because we may have situations\n    where idx > size due to the way indices are calculated\n    during slice/range analysis.\n\n    Both idx and size have to be Integer types.\n    size should be from the array size vars that array_analysis\n    adds and the bitwidth should match the platform maximum.\n    '
    require(isinstance(idx, types.scalars.Integer))
    require(isinstance(size, types.scalars.Integer))
    unified_ty = types.intp if size.signed else types.uintp
    idx_unified = types.intp if idx.signed else types.uintp

    def codegen(context, builder, sig, args):
        if False:
            while True:
                i = 10
        ll_idx_unified_ty = context.get_data_type(idx_unified)
        ll_unified_ty = context.get_data_type(unified_ty)
        if idx_unified.signed:
            idx = builder.sext(args[0], ll_idx_unified_ty)
        else:
            idx = builder.zext(args[0], ll_idx_unified_ty)
        if unified_ty.signed:
            size = builder.sext(args[1], ll_unified_ty)
        else:
            size = builder.zext(args[1], ll_unified_ty)
        neg_size = builder.neg(size)
        zero = llvmlite.ir.Constant(ll_unified_ty, 0)
        idx_negative = builder.icmp_signed('<', idx, zero)
        pos_oversize = builder.icmp_signed('>=', idx, size)
        neg_oversize = builder.icmp_signed('<=', idx, neg_size)
        pos_res = builder.select(pos_oversize, size, idx)
        neg_res = builder.select(neg_oversize, zero, builder.add(idx, size))
        mod = builder.select(idx_negative, neg_res, pos_res)
        return mod
    return (signature(unified_ty, idx, size), codegen)

def wrap_index_literal(idx, size):
    if False:
        i = 10
        return i + 15
    if idx < 0:
        if idx <= -size:
            return 0
        else:
            return idx + size
    elif idx >= size:
        return size
    else:
        return idx

@intrinsic
def assert_equiv(typingctx, *val):
    if False:
        for i in range(10):
            print('nop')
    '\n    A function that asserts the inputs are of equivalent size,\n    and throws runtime error when they are not. The input is\n    a vararg that contains an error message, followed by a set\n    of objects of either array, tuple or integer.\n    '
    if len(val) > 1:
        val = (types.StarArgTuple(val),)
    assert len(val[0]) > 1
    assert all((isinstance(a, (types.ArrayCompatible, types.BaseTuple, types.SliceType, types.Integer)) for a in val[0][1:]))
    if not isinstance(val[0][0], types.StringLiteral):
        raise errors.TypingError('first argument must be a StringLiteral')

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        assert len(args) == 1
        tup = cgutils.unpack_tuple(builder, args[0])
        tup_type = sig.args[0]
        msg = sig.args[0][0].literal_value

        def unpack_shapes(a, aty):
            if False:
                return 10
            if isinstance(aty, types.ArrayCompatible):
                ary = context.make_array(aty)(context, builder, a)
                return cgutils.unpack_tuple(builder, ary.shape)
            elif isinstance(aty, types.BaseTuple):
                return cgutils.unpack_tuple(builder, a)
            else:
                return [a]

        def pairwise(a, aty, b, bty):
            if False:
                while True:
                    i = 10
            ashapes = unpack_shapes(a, aty)
            bshapes = unpack_shapes(b, bty)
            assert len(ashapes) == len(bshapes)
            for (m, n) in zip(ashapes, bshapes):
                m_eq_n = builder.icmp_unsigned('==', m, n)
                with builder.if_else(m_eq_n) as (then, orelse):
                    with then:
                        pass
                    with orelse:
                        context.call_conv.return_user_exc(builder, AssertionError, (msg,))
        for i in range(1, len(tup_type) - 1):
            pairwise(tup[i], tup_type[i], tup[i + 1], tup_type[i + 1])
        r = context.get_constant_generic(builder, types.NoneType, None)
        return r
    return (signature(types.none, *val), codegen)

class EquivSet(object):
    """EquivSet keeps track of equivalence relations between
    a set of objects.
    """

    def __init__(self, obj_to_ind=None, ind_to_obj=None, next_ind=0):
        if False:
            while True:
                i = 10
        'Create a new EquivSet object. Optional keyword arguments are for\n        internal use only.\n        '
        self.obj_to_ind = obj_to_ind if obj_to_ind else {}
        self.ind_to_obj = ind_to_obj if ind_to_obj else {}
        self.next_ind = next_ind

    def empty(self):
        if False:
            while True:
                i = 10
        'Return an empty EquivSet object.\n        '
        return EquivSet()

    def clone(self):
        if False:
            while True:
                i = 10
        'Return a new copy.\n        '
        return EquivSet(obj_to_ind=copy.deepcopy(self.obj_to_ind), ind_to_obj=copy.deepcopy(self.ind_to_obj), next_id=self.next_ind)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'EquivSet({})'.format(self.ind_to_obj)

    def is_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Return true if the set is empty, or false otherwise.\n        '
        return self.obj_to_ind == {}

    def _get_ind(self, x):
        if False:
            print('Hello World!')
        'Return the internal index (greater or equal to 0) of the given\n        object, or -1 if not found.\n        '
        return self.obj_to_ind.get(x, -1)

    def _get_or_add_ind(self, x):
        if False:
            return 10
        'Return the internal index (greater or equal to 0) of the given\n        object, or create a new one if not found.\n        '
        if x in self.obj_to_ind:
            i = self.obj_to_ind[x]
        else:
            i = self.next_ind
            self.next_ind += 1
        return i

    def _insert(self, objs):
        if False:
            i = 10
            return i + 15
        'Base method that inserts a set of equivalent objects by modifying\n        self.\n        '
        assert len(objs) > 1
        inds = tuple((self._get_or_add_ind(x) for x in objs))
        ind = min(inds)
        if config.DEBUG_ARRAY_OPT >= 2:
            print('_insert:', objs, inds)
        if not ind in self.ind_to_obj:
            self.ind_to_obj[ind] = []
        for (i, obj) in zip(inds, objs):
            if i == ind:
                if not obj in self.ind_to_obj[ind]:
                    self.ind_to_obj[ind].append(obj)
                    self.obj_to_ind[obj] = ind
            elif i in self.ind_to_obj:
                for x in self.ind_to_obj[i]:
                    self.obj_to_ind[x] = ind
                    self.ind_to_obj[ind].append(x)
                del self.ind_to_obj[i]
            else:
                self.obj_to_ind[obj] = ind
                self.ind_to_obj[ind].append(obj)

    def is_equiv(self, *objs):
        if False:
            i = 10
            return i + 15
        'Try to derive if given objects are equivalent, return true\n        if so, or false otherwise.\n        '
        inds = [self._get_ind(x) for x in objs]
        ind = max(inds)
        if ind != -1:
            return all((i == ind for i in inds))
        else:
            return all([x == objs[0] for x in objs])

    def get_equiv_const(self, obj):
        if False:
            i = 10
            return i + 15
        'Check if obj is equivalent to some int constant, and return\n        the constant if found, or None otherwise.\n        '
        ind = self._get_ind(obj)
        if ind >= 0:
            objs = self.ind_to_obj[ind]
            for x in objs:
                if isinstance(x, int):
                    return x
        return None

    def get_equiv_set(self, obj):
        if False:
            return 10
        'Return the set of equivalent objects.\n        '
        ind = self._get_ind(obj)
        if ind >= 0:
            return set(self.ind_to_obj[ind])
        return set()

    def insert_equiv(self, *objs):
        if False:
            print('Hello World!')
        'Insert a set of equivalent objects by modifying self. This\n        method can be overloaded to transform object type before insertion.\n        '
        return self._insert(objs)

    def intersect(self, equiv_set):
        if False:
            return 10
        ' Return the intersection of self and the given equiv_set,\n        without modifying either of them. The result will also keep\n        old equivalence indices unchanged.\n        '
        new_set = self.empty()
        new_set.next_ind = self.next_ind
        for objs in equiv_set.ind_to_obj.values():
            inds = tuple((self._get_ind(x) for x in objs))
            ind_to_obj = {}
            for (i, x) in zip(inds, objs):
                if i in ind_to_obj:
                    ind_to_obj[i].append(x)
                elif i >= 0:
                    ind_to_obj[i] = [x]
            for v in ind_to_obj.values():
                if len(v) > 1:
                    new_set._insert(v)
        return new_set

class ShapeEquivSet(EquivSet):
    """Just like EquivSet, except that it accepts only numba IR variables
    and constants as objects, guided by their types. Arrays are considered
    equivalent as long as their shapes are equivalent. Scalars are
    equivalent only when they are equal in value. Tuples are equivalent
    when they are of the same size, and their elements are equivalent.
    """

    def __init__(self, typemap, defs=None, ind_to_var=None, obj_to_ind=None, ind_to_obj=None, next_id=0, ind_to_const=None):
        if False:
            while True:
                i = 10
        'Create a new ShapeEquivSet object, where typemap is a dictionary\n        that maps variable names to their types, and it will not be modified.\n        Optional keyword arguments are for internal use only.\n        '
        self.typemap = typemap
        self.defs = defs if defs else {}
        self.ind_to_var = ind_to_var if ind_to_var else {}
        self.ind_to_const = ind_to_const if ind_to_const else {}
        super(ShapeEquivSet, self).__init__(obj_to_ind, ind_to_obj, next_id)

    def empty(self):
        if False:
            print('Hello World!')
        'Return an empty ShapeEquivSet.\n        '
        return ShapeEquivSet(self.typemap, {})

    def clone(self):
        if False:
            return 10
        'Return a new copy.\n        '
        return ShapeEquivSet(self.typemap, defs=copy.copy(self.defs), ind_to_var=copy.copy(self.ind_to_var), obj_to_ind=copy.deepcopy(self.obj_to_ind), ind_to_obj=copy.deepcopy(self.ind_to_obj), next_id=self.next_ind, ind_to_const=copy.deepcopy(self.ind_toconst))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ShapeEquivSet({}, ind_to_var={}, ind_to_const={})'.format(self.ind_to_obj, self.ind_to_var, self.ind_to_const)

    def _get_names(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Return a set of names for the given obj, where array and tuples\n        are broken down to their individual shapes or elements. This is\n        safe because both Numba array shapes and Python tuples are immutable.\n        '
        if isinstance(obj, ir.Var) or isinstance(obj, str):
            name = obj if isinstance(obj, str) else obj.name
            if name not in self.typemap:
                return (name,)
            typ = self.typemap[name]
            if isinstance(typ, (types.BaseTuple, types.ArrayCompatible)):
                ndim = typ.ndim if isinstance(typ, types.ArrayCompatible) else len(typ)
                if ndim == 0:
                    return (name,)
                else:
                    return tuple(('{}#{}'.format(name, i) for i in range(ndim)))
            else:
                return (name,)
        elif isinstance(obj, ir.Const):
            if isinstance(obj.value, tuple):
                return obj.value
            else:
                return (obj.value,)
        elif isinstance(obj, tuple):

            def get_names(x):
                if False:
                    for i in range(10):
                        print('nop')
                names = self._get_names(x)
                if len(names) != 0:
                    return names[0]
                return names
            return tuple((get_names(x) for x in obj))
        elif isinstance(obj, int):
            return (obj,)
        if config.DEBUG_ARRAY_OPT >= 1:
            print(f'Ignoring untracked object type {type(obj)} in ShapeEquivSet')
        return ()

    def is_equiv(self, *objs):
        if False:
            return 10
        'Overload EquivSet.is_equiv to handle Numba IR variables and\n        constants.\n        '
        assert len(objs) > 1
        obj_names = [self._get_names(x) for x in objs]
        obj_names = [x for x in obj_names if x != ()]
        if len(obj_names) <= 1:
            return False
        ndims = [len(names) for names in obj_names]
        ndim = ndims[0]
        if not all((ndim == x for x in ndims)):
            if config.DEBUG_ARRAY_OPT >= 1:
                print('is_equiv: Dimension mismatch for {}'.format(objs))
            return False
        for i in range(ndim):
            names = [obj_name[i] for obj_name in obj_names]
            if not super(ShapeEquivSet, self).is_equiv(*names):
                return False
        return True

    def get_equiv_const(self, obj):
        if False:
            while True:
                i = 10
        'If the given object is equivalent to a constant scalar,\n        return the scalar value, or None otherwise.\n        '
        names = self._get_names(obj)
        if len(names) != 1:
            return None
        return super(ShapeEquivSet, self).get_equiv_const(names[0])

    def get_equiv_var(self, obj):
        if False:
            i = 10
            return i + 15
        'If the given object is equivalent to some defined variable,\n        return the variable, or None otherwise.\n        '
        names = self._get_names(obj)
        if len(names) != 1:
            return None
        ind = self._get_ind(names[0])
        vs = self.ind_to_var.get(ind, [])
        return vs[0] if vs != [] else None

    def get_equiv_set(self, obj):
        if False:
            i = 10
            return i + 15
        'Return the set of equivalent objects.\n        '
        names = self._get_names(obj)
        if len(names) != 1:
            return None
        return super(ShapeEquivSet, self).get_equiv_set(names[0])

    def _insert(self, objs):
        if False:
            print('Hello World!')
        'Overload EquivSet._insert to manage ind_to_var dictionary.\n        '
        inds = []
        for obj in objs:
            if obj in self.obj_to_ind:
                inds.append(self.obj_to_ind[obj])
        varlist = []
        constval = None
        names = set()
        for i in sorted(inds):
            if i in self.ind_to_var:
                for x in self.ind_to_var[i]:
                    if not x.name in names:
                        varlist.append(x)
                        names.add(x.name)
            if i in self.ind_to_const:
                assert constval is None
                constval = self.ind_to_const[i]
        super(ShapeEquivSet, self)._insert(objs)
        new_ind = self.obj_to_ind[objs[0]]
        for i in set(inds):
            if i in self.ind_to_var:
                del self.ind_to_var[i]
        self.ind_to_var[new_ind] = varlist
        if constval is not None:
            self.ind_to_const[new_ind] = constval

    def insert_equiv(self, *objs):
        if False:
            for i in range(10):
                print('nop')
        'Overload EquivSet.insert_equiv to handle Numba IR variables and\n        constants. Input objs are either variable or constant, and at least\n        one of them must be variable.\n        '
        assert len(objs) > 1
        obj_names = [self._get_names(x) for x in objs]
        obj_names = [x for x in obj_names if x != ()]
        if len(obj_names) <= 1:
            return
        names = sum([list(x) for x in obj_names], [])
        ndims = [len(x) for x in obj_names]
        ndim = ndims[0]
        assert all((ndim == x for x in ndims)), 'Dimension mismatch for {}'.format(objs)
        varlist = []
        constlist = []
        for obj in objs:
            if not isinstance(obj, tuple):
                obj = (obj,)
            for var in obj:
                if isinstance(var, ir.Var) and (not var.name in varlist):
                    if var.name in self.defs:
                        varlist.insert(0, var)
                    else:
                        varlist.append(var)
                if isinstance(var, ir.Const) and (not var.value in constlist):
                    constlist.append(var.value)
        for obj in varlist:
            name = obj.name
            if name in names and (not name in self.obj_to_ind):
                self.ind_to_obj[self.next_ind] = [name]
                self.obj_to_ind[name] = self.next_ind
                self.ind_to_var[self.next_ind] = [obj]
                self.next_ind += 1
        for const in constlist:
            if const in names and (not const in self.obj_to_ind):
                self.ind_to_obj[self.next_ind] = [const]
                self.obj_to_ind[const] = self.next_ind
                self.ind_to_const[self.next_ind] = const
                self.next_ind += 1
        some_change = False
        for i in range(ndim):
            names = [obj_name[i] for obj_name in obj_names]
            ie_res = super(ShapeEquivSet, self).insert_equiv(*names)
            some_change = some_change or ie_res
        return some_change

    def has_shape(self, name):
        if False:
            return 10
        'Return true if the shape of the given variable is available.\n        '
        return self.get_shape(name) is not None

    def get_shape(self, name):
        if False:
            while True:
                i = 10
        'Return a tuple of variables that corresponds to the shape\n        of the given array, or None if not found.\n        '
        return guard(self._get_shape, name)

    def _get_shape(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple of variables that corresponds to the shape\n        of the given array, or raise GuardException if not found.\n        '
        inds = self.get_shape_classes(name)
        require(inds != ())
        shape = []
        for i in inds:
            require(i in self.ind_to_var)
            vs = self.ind_to_var[i]
            if vs != []:
                shape.append(vs[0])
            else:
                require(i in self.ind_to_const)
                vs = self.ind_to_const[i]
                shape.append(vs)
        return tuple(shape)

    def get_shape_classes(self, name):
        if False:
            i = 10
            return i + 15
        'Instead of the shape tuple, return tuple of int, where\n        each int is the corresponding class index of the size object.\n        Unknown shapes are given class index -1. Return empty tuple\n        if the input name is a scalar variable.\n        '
        if isinstance(name, ir.Var):
            name = name.name
        typ = self.typemap[name] if name in self.typemap else None
        if not isinstance(typ, (types.BaseTuple, types.SliceType, types.ArrayCompatible)):
            return []
        if isinstance(typ, types.ArrayCompatible) and typ.ndim == 0:
            return []
        names = self._get_names(name)
        inds = tuple((self._get_ind(name) for name in names))
        return inds

    def intersect(self, equiv_set):
        if False:
            return 10
        'Overload the intersect method to handle ind_to_var.\n        '
        newset = super(ShapeEquivSet, self).intersect(equiv_set)
        ind_to_var = {}
        for (i, objs) in newset.ind_to_obj.items():
            assert len(objs) > 0
            obj = objs[0]
            assert obj in self.obj_to_ind
            assert obj in equiv_set.obj_to_ind
            j = self.obj_to_ind[obj]
            k = equiv_set.obj_to_ind[obj]
            assert j in self.ind_to_var
            assert k in equiv_set.ind_to_var
            varlist = []
            names = [x.name for x in equiv_set.ind_to_var[k]]
            for x in self.ind_to_var[j]:
                if x.name in names:
                    varlist.append(x)
            ind_to_var[i] = varlist
        newset.ind_to_var = ind_to_var
        return newset

    def define(self, name, redefined):
        if False:
            i = 10
            return i + 15
        "Increment the internal count of how many times a variable is being\n        defined. Most variables in Numba IR are SSA, i.e., defined only once,\n        but not all of them. When a variable is being re-defined, it must\n        be removed from the equivalence relation and added to the redefined\n        set but only if that redefinition is not known to have the same\n        equivalence classes. Those variables redefined are removed from all\n        the blocks' equivalence sets later.\n\n        Arrays passed to define() use their whole name but these do not\n        appear in the equivalence sets since they are stored there per\n        dimension. Calling _get_names() here converts array names to\n        dimensional names.\n\n        This function would previously invalidate if there were any multiple\n        definitions of a variable.  However, we realized that this behavior\n        is overly restrictive.  You need only invalidate on multiple\n        definitions if they are not known to be equivalent. So, the\n        equivalence insertion functions now return True if some change was\n        made (meaning the definition was not equivalent) and False\n        otherwise. If no change was made, then define() need not be\n        called. For no change to have been made, the variable must\n        already be present. If the new definition of the var has the\n        case where lhs and rhs are in the same equivalence class then\n        again, no change will be made and define() need not be called\n        or the variable invalidated.\n        "
        if isinstance(name, ir.Var):
            name = name.name
        if name in self.defs:
            self.defs[name] += 1
            name_res = list(self._get_names(name))
            for one_name in name_res:
                if one_name in self.obj_to_ind:
                    redefined.add(one_name)
                    i = self.obj_to_ind[one_name]
                    del self.obj_to_ind[one_name]
                    self.ind_to_obj[i].remove(one_name)
                    if self.ind_to_obj[i] == []:
                        del self.ind_to_obj[i]
                    assert i in self.ind_to_var
                    names = [x.name for x in self.ind_to_var[i]]
                    if name in names:
                        j = names.index(name)
                        del self.ind_to_var[i][j]
                        if self.ind_to_var[i] == []:
                            del self.ind_to_var[i]
                            if i in self.ind_to_obj:
                                for obj in self.ind_to_obj[i]:
                                    del self.obj_to_ind[obj]
                                del self.ind_to_obj[i]
        else:
            self.defs[name] = 1

    def union_defs(self, defs, redefined):
        if False:
            print('Hello World!')
        'Union with the given defs dictionary. This is meant to handle\n        branch join-point, where a variable may have been defined in more\n        than one branches.\n        '
        for (k, v) in defs.items():
            if v > 0:
                self.define(k, redefined)

class SymbolicEquivSet(ShapeEquivSet):
    """Just like ShapeEquivSet, except that it also reasons about variable
    equivalence symbolically by using their arithmetic definitions.
    The goal is to automatically derive the equivalence of array ranges
    (slicing). For instance, a[1:m] and a[0:m-1] shall be considered
    size-equivalence.
    """

    def __init__(self, typemap, def_by=None, ref_by=None, ext_shapes=None, defs=None, ind_to_var=None, obj_to_ind=None, ind_to_obj=None, next_id=0):
        if False:
            for i in range(10):
                print('nop')
        'Create a new SymbolicEquivSet object, where typemap is a dictionary\n        that maps variable names to their types, and it will not be modified.\n        Optional keyword arguments are for internal use only.\n        '
        self.def_by = def_by if def_by else {}
        self.ref_by = ref_by if ref_by else {}
        self.ext_shapes = ext_shapes if ext_shapes else {}
        self.rel_map = {}
        self.wrap_map = {}
        super(SymbolicEquivSet, self).__init__(typemap, defs, ind_to_var, obj_to_ind, ind_to_obj, next_id)

    def empty(self):
        if False:
            while True:
                i = 10
        'Return an empty SymbolicEquivSet.\n        '
        return SymbolicEquivSet(self.typemap)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'SymbolicEquivSet({}, ind_to_var={}, def_by={}, ref_by={}, ext_shapes={})'.format(self.ind_to_obj, self.ind_to_var, self.def_by, self.ref_by, self.ext_shapes)

    def clone(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a new copy.\n        '
        return SymbolicEquivSet(self.typemap, def_by=copy.copy(self.def_by), ref_by=copy.copy(self.ref_by), ext_shapes=copy.copy(self.ext_shapes), defs=copy.copy(self.defs), ind_to_var=copy.copy(self.ind_to_var), obj_to_ind=copy.deepcopy(self.obj_to_ind), ind_to_obj=copy.deepcopy(self.ind_to_obj), next_id=self.next_ind)

    def get_rel(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve a definition pair for the given variable,\n        or return None if it is not available.\n        '
        return guard(self._get_or_set_rel, name)

    def _get_or_set_rel(self, name, func_ir=None):
        if False:
            print('Hello World!')
        'Retrieve a definition pair for the given variable,\n        and if it is not already available, try to look it up\n        in the given func_ir, and remember it for future use.\n        '
        if isinstance(name, ir.Var):
            name = name.name
        require(self.defs.get(name, 0) == 1)
        if name in self.def_by:
            return self.def_by[name]
        else:
            require(func_ir is not None)

            def plus(x, y):
                if False:
                    return 10
                x_is_const = isinstance(x, int)
                y_is_const = isinstance(y, int)
                if x_is_const:
                    if y_is_const:
                        return x + y
                    else:
                        (var, offset) = y
                        return (var, x + offset)
                else:
                    (var, offset) = x
                    if y_is_const:
                        return (var, y + offset)
                    else:
                        return None

            def minus(x, y):
                if False:
                    while True:
                        i = 10
                if isinstance(y, int):
                    return plus(x, -y)
                elif isinstance(x, tuple) and isinstance(y, tuple) and (x[0] == y[0]):
                    return minus(x[1], y[1])
                else:
                    return None
            expr = get_definition(func_ir, name)
            value = (name, 0)
            if isinstance(expr, ir.Expr):
                if expr.op == 'call':
                    (fname, mod_name) = find_callname(func_ir, expr, typemap=self.typemap)
                    if fname == 'wrap_index' and mod_name == 'numba.parfors.array_analysis':
                        index = tuple((self.obj_to_ind.get(x.name, -1) for x in expr.args))
                        if -1 in index:
                            return None
                        names = self.ext_shapes.get(index, [])
                        names.append(name)
                        if len(names) > 0:
                            self._insert(names)
                        self.ext_shapes[index] = names
                elif expr.op == 'binop':
                    lhs = self._get_or_set_rel(expr.lhs, func_ir)
                    rhs = self._get_or_set_rel(expr.rhs, func_ir)
                    if lhs is None or rhs is None:
                        return None
                    elif expr.fn == operator.add:
                        value = plus(lhs, rhs)
                    elif expr.fn == operator.sub:
                        value = minus(lhs, rhs)
            elif isinstance(expr, ir.Const) and isinstance(expr.value, int):
                value = expr.value
            require(value is not None)
            self.def_by[name] = value
            if isinstance(value, int) or (isinstance(value, tuple) and (value[0] != name or value[1] != 0)):
                if isinstance(value, tuple):
                    (var, offset) = value
                    if not var in self.ref_by:
                        self.ref_by[var] = []
                    self.ref_by[var].append((name, -offset))
                    ind = self._get_ind(var)
                    if ind >= 0:
                        objs = self.ind_to_obj[ind]
                        names = []
                        for obj in objs:
                            if obj in self.ref_by:
                                names += [x for (x, i) in self.ref_by[obj] if i == -offset]
                        if len(names) > 1:
                            super(SymbolicEquivSet, self)._insert(names)
            return value

    def define(self, var, redefined, func_ir=None, typ=None):
        if False:
            return 10
        'Besides incrementing the definition count of the given variable\n        name, it will also retrieve and simplify its definition from func_ir,\n        and remember the result for later equivalence comparison. Supported\n        operations are:\n          1. arithmetic plus and minus with constants\n          2. wrap_index (relative to some given size)\n        '
        if isinstance(var, ir.Var):
            name = var.name
        else:
            name = var
        super(SymbolicEquivSet, self).define(name, redefined)
        if func_ir and self.defs.get(name, 0) == 1 and isinstance(typ, types.Number):
            value = guard(self._get_or_set_rel, name, func_ir)
            if isinstance(value, int):
                self._insert([name, value])
            if isinstance(var, ir.Var):
                ind = self._get_or_add_ind(name)
                if not ind in self.ind_to_obj:
                    self.ind_to_obj[ind] = [name]
                    self.obj_to_ind[name] = ind
                if ind in self.ind_to_var:
                    self.ind_to_var[ind].append(var)
                else:
                    self.ind_to_var[ind] = [var]
            return True

    def _insert(self, objs):
        if False:
            for i in range(10):
                print('nop')
        'Overload _insert method to handle ind changes between relative\n        objects.  Returns True if some change is made, false otherwise.\n        '
        indset = set()
        uniqs = set()
        for obj in objs:
            ind = self._get_ind(obj)
            if ind == -1:
                uniqs.add(obj)
            elif not ind in indset:
                uniqs.add(obj)
                indset.add(ind)
        if len(uniqs) <= 1:
            return False
        uniqs = list(uniqs)
        super(SymbolicEquivSet, self)._insert(uniqs)
        objs = self.ind_to_obj[self._get_ind(uniqs[0])]
        offset_dict = {}

        def get_or_set(d, k):
            if False:
                return 10
            if k in d:
                v = d[k]
            else:
                v = []
                d[k] = v
            return v
        for obj in objs:
            if obj in self.def_by:
                value = self.def_by[obj]
                if isinstance(value, tuple):
                    (name, offset) = value
                    get_or_set(offset_dict, -offset).append(name)
                    if name in self.ref_by:
                        for (v, i) in self.ref_by[name]:
                            get_or_set(offset_dict, -(offset + i)).append(v)
            if obj in self.ref_by:
                for (name, offset) in self.ref_by[obj]:
                    get_or_set(offset_dict, offset).append(name)
        for names in offset_dict.values():
            self._insert(names)
        return True

    def set_shape_setitem(self, obj, shape):
        if False:
            for i in range(10):
                print('nop')
        'remember shapes of SetItem IR nodes.\n        '
        assert isinstance(obj, (ir.StaticSetItem, ir.SetItem))
        self.ext_shapes[obj] = shape

    def _get_shape(self, obj):
        if False:
            i = 10
            return i + 15
        'Overload _get_shape to retrieve the shape of SetItem IR nodes.\n        '
        if isinstance(obj, (ir.StaticSetItem, ir.SetItem)):
            require(obj in self.ext_shapes)
            return self.ext_shapes[obj]
        else:
            assert isinstance(obj, ir.Var)
            typ = self.typemap[obj.name]
            if isinstance(typ, types.SliceType):
                return (obj,)
            else:
                return super(SymbolicEquivSet, self)._get_shape(obj)

class WrapIndexMeta(object):
    """
      Array analysis should be able to analyze all the function
      calls that it adds to the IR.  That way, array analysis can
      be run as often as needed and you should get the same
      equivalencies.  One modification to the IR that array analysis
      makes is the insertion of wrap_index calls.  Thus, repeated
      array analysis passes should be able to analyze these wrap_index
      calls.  The difficulty of these calls is that the equivalence
      class of the left-hand side of the assignment is not present in
      the arguments to wrap_index in the right-hand side.  Instead,
      the equivalence class of the wrap_index output is a combination
      of the wrap_index args.  The important thing to
      note is that if the equivalence classes of the slice size
      and the dimension's size are the same for two wrap index
      calls then we can be assured of the answer being the same.
      So, we maintain the wrap_map dict that maps from a tuple
      of equivalence class ids for the slice and dimension size
      to some new equivalence class id for the output size.
      However, when we are analyzing the first such wrap_index
      call we don't have a variable there to associate to the
      size since we're in the process of analyzing the instruction
      that creates that mapping.  So, instead we return an object
      of this special class and analyze_inst will establish the
      connection between a tuple of the parts of this object
      below and the left-hand side variable.
    """

    def __init__(self, slice_size, dim_size):
        if False:
            for i in range(10):
                print('nop')
        self.slice_size = slice_size
        self.dim_size = dim_size

class ArrayAnalysis(object):
    aa_count = 0
    'Analyzes Numpy array computations for properties such as\n    shape/size equivalence, and keeps track of them on a per-block\n    basis. The analysis should only be run once because it modifies\n    the incoming IR by inserting assertion statements that safeguard\n    parfor optimizations.\n    '

    def __init__(self, context, func_ir, typemap, calltypes):
        if False:
            i = 10
            return i + 15
        self.context = context
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.equiv_sets = {}
        self.array_attr_calls = {}
        self.object_attrs = {}
        self.prepends = {}
        self.pruned_predecessors = {}

    def get_equiv_set(self, block_label):
        if False:
            for i in range(10):
                print('nop')
        'Return the equiv_set object of an block given its label.\n        '
        return self.equiv_sets[block_label]

    def remove_redefineds(self, redefineds):
        if False:
            while True:
                i = 10
        'Take a set of variables in redefineds and go through all\n        the currently existing equivalence sets (created in topo order)\n        and remove that variable from all of them since it is multiply\n        defined within the function.\n        '
        unused = set()
        for r in redefineds:
            for eslabel in self.equiv_sets:
                es = self.equiv_sets[eslabel]
                es.define(r, unused)

    def run(self, blocks=None, equiv_set=None):
        if False:
            while True:
                i = 10
        'run array shape analysis on the given IR blocks, resulting in\n        modified IR and finalized EquivSet for each block.\n        '
        if blocks is None:
            blocks = self.func_ir.blocks
        self.func_ir._definitions = build_definitions(self.func_ir.blocks)
        if equiv_set is None:
            init_equiv_set = SymbolicEquivSet(self.typemap)
        else:
            init_equiv_set = equiv_set
        (self.alias_map, self.arg_aliases) = find_potential_aliases(blocks, self.func_ir.arg_names, self.typemap, self.func_ir)
        aa_count_save = ArrayAnalysis.aa_count
        ArrayAnalysis.aa_count += 1
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Starting ArrayAnalysis:', aa_count_save)
        dprint_func_ir(self.func_ir, 'before array analysis', blocks)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('ArrayAnalysis variable types: ', sorted(self.typemap.items()))
            print('ArrayAnalysis call types: ', self.calltypes)
        cfg = compute_cfg_from_blocks(blocks)
        topo_order = find_topo_order(blocks, cfg=cfg)
        self._run_on_blocks(topo_order, blocks, cfg, init_equiv_set)
        if config.DEBUG_ARRAY_OPT >= 1:
            self.dump()
            print('ArrayAnalysis post variable types: ', sorted(self.typemap.items()))
            print('ArrayAnalysis post call types: ', self.calltypes)
        dprint_func_ir(self.func_ir, 'after array analysis', blocks)
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Ending ArrayAnalysis:', aa_count_save)

    def _run_on_blocks(self, topo_order, blocks, cfg, init_equiv_set):
        if False:
            print('Hello World!')
        for label in topo_order:
            if config.DEBUG_ARRAY_OPT >= 2:
                print('Processing block:', label)
            block = blocks[label]
            scope = block.scope
            pending_transforms = self._determine_transform(cfg, block, label, scope, init_equiv_set)
            self._combine_to_new_block(block, pending_transforms)

    def _combine_to_new_block(self, block, pending_transforms):
        if False:
            while True:
                i = 10
        'Combine the new instructions from previous pass into a new block\n        body.\n        '
        new_body = []
        for (inst, pre, post) in pending_transforms:
            for instr in pre:
                new_body.append(instr)
            new_body.append(inst)
            for instr in post:
                new_body.append(instr)
        block.body = new_body

    def _determine_transform(self, cfg, block, label, scope, init_equiv_set):
        if False:
            print('Hello World!')
        'Determine the transformation for each instruction in the block\n        '
        equiv_set = None
        preds = cfg.predecessors(label)
        if label in self.pruned_predecessors:
            pruned = self.pruned_predecessors[label]
        else:
            pruned = []
        if config.DEBUG_ARRAY_OPT >= 2:
            print('preds:', preds)
        for (p, q) in preds:
            if config.DEBUG_ARRAY_OPT >= 2:
                print('p, q:', p, q)
            if p in pruned:
                continue
            if p in self.equiv_sets:
                from_set = self.equiv_sets[p].clone()
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('p in equiv_sets', from_set)
                if (p, label) in self.prepends:
                    instrs = self.prepends[p, label]
                    for inst in instrs:
                        redefined = set()
                        self._analyze_inst(label, scope, from_set, inst, redefined)
                        self.remove_redefineds(redefined)
                if equiv_set is None:
                    equiv_set = from_set
                else:
                    equiv_set = equiv_set.intersect(from_set)
                    redefined = set()
                    equiv_set.union_defs(from_set.defs, redefined)
                    self.remove_redefineds(redefined)
        if equiv_set is None:
            equiv_set = init_equiv_set
        self.equiv_sets[label] = equiv_set
        pending_transforms = []
        for inst in block.body:
            redefined = set()
            (pre, post) = self._analyze_inst(label, scope, equiv_set, inst, redefined)
            if len(redefined) > 0:
                self.remove_redefineds(redefined)
            pending_transforms.append((inst, pre, post))
        return pending_transforms

    def dump(self):
        if False:
            for i in range(10):
                print('nop')
        'dump per-block equivalence sets for debugging purposes.\n        '
        print('Array Analysis: ', self.equiv_sets)

    def _define(self, equiv_set, var, typ, value):
        if False:
            return 10
        self.typemap[var.name] = typ
        self.func_ir._definitions[var.name] = [value]
        redefineds = set()
        equiv_set.define(var, redefineds, self.func_ir, typ)

    class AnalyzeResult(object):

        def __init__(self, **kwargs):
            if False:
                i = 10
                return i + 15
            self.kwargs = kwargs

    def _analyze_inst(self, label, scope, equiv_set, inst, redefined):
        if False:
            while True:
                i = 10
        pre = []
        post = []
        if config.DEBUG_ARRAY_OPT >= 2:
            print('analyze_inst:', inst)
        if isinstance(inst, ir.Assign):
            lhs = inst.target
            typ = self.typemap[lhs.name]
            shape = None
            if isinstance(typ, types.ArrayCompatible) and typ.ndim == 0:
                shape = ()
            elif isinstance(inst.value, ir.Expr):
                result = self._analyze_expr(scope, equiv_set, inst.value, lhs)
                if result:
                    require(isinstance(result, ArrayAnalysis.AnalyzeResult))
                    if 'shape' in result.kwargs:
                        shape = result.kwargs['shape']
                    if 'pre' in result.kwargs:
                        pre.extend(result.kwargs['pre'])
                    if 'post' in result.kwargs:
                        post.extend(result.kwargs['post'])
                    if 'rhs' in result.kwargs:
                        inst.value = result.kwargs['rhs']
            elif isinstance(inst.value, (ir.Var, ir.Const)):
                shape = inst.value
            elif isinstance(inst.value, ir.Global):
                gvalue = inst.value.value
                if isinstance(gvalue, tuple) and all((isinstance(v, int) for v in gvalue)):
                    shape = gvalue
                elif isinstance(gvalue, int):
                    shape = (gvalue,)
            elif isinstance(inst.value, ir.Arg):
                if isinstance(typ, types.containers.UniTuple) and isinstance(typ.dtype, types.Integer):
                    shape = inst.value
                elif isinstance(typ, types.containers.Tuple) and all([isinstance(x, (types.Integer, types.IntegerLiteral)) for x in typ.types]):
                    shape = inst.value
            if isinstance(shape, ir.Const):
                if isinstance(shape.value, tuple):
                    loc = shape.loc
                    shape = tuple((ir.Const(x, loc) for x in shape.value))
                elif isinstance(shape.value, int):
                    shape = (shape,)
                else:
                    shape = None
            elif isinstance(shape, ir.Var) and isinstance(self.typemap[shape.name], types.Integer):
                shape = (shape,)
            elif isinstance(shape, WrapIndexMeta):
                " Here we've got the special WrapIndexMeta object\n                    back from analyzing a wrap_index call.  We define\n                    the lhs and then get it's equivalence class then\n                    add the mapping from the tuple of slice size and\n                    dimensional size equivalence ids to the lhs\n                    equivalence id.\n                "
                equiv_set.define(lhs, redefined, self.func_ir, typ)
                lhs_ind = equiv_set._get_ind(lhs.name)
                if lhs_ind != -1:
                    equiv_set.wrap_map[shape.slice_size, shape.dim_size] = lhs_ind
                return (pre, post)
            if isinstance(typ, types.ArrayCompatible):
                if shape is not None and isinstance(shape, ir.Var) and isinstance(self.typemap[shape.name], types.containers.BaseTuple):
                    pass
                elif shape is None or isinstance(shape, tuple) or (isinstance(shape, ir.Var) and (not equiv_set.has_shape(shape))):
                    shape = self._gen_shape_call(equiv_set, lhs, typ.ndim, shape, post)
            elif isinstance(typ, types.UniTuple):
                if shape and isinstance(typ.dtype, types.Integer):
                    shape = self._gen_shape_call(equiv_set, lhs, len(typ), shape, post)
            elif isinstance(typ, types.containers.Tuple) and all([isinstance(x, (types.Integer, types.IntegerLiteral)) for x in typ.types]):
                shape = self._gen_shape_call(equiv_set, lhs, len(typ), shape, post)
            " See the comment on the define() function.\n\n                We need only call define(), which will invalidate a variable\n                from being in the equivalence sets on multiple definitions,\n                if the variable was not previously defined or if the new\n                definition would be in a conflicting equivalence class to the\n                original equivalence class for the variable.\n\n                insert_equiv() returns True if either of these conditions are\n                True and then we call define() in those cases.\n                If insert_equiv() returns False then no changes were made and\n                all equivalence classes are consistent upon a redefinition so\n                no invalidation is needed and we don't call define().\n            "
            needs_define = True
            if shape is not None:
                needs_define = equiv_set.insert_equiv(lhs, shape)
            if needs_define:
                equiv_set.define(lhs, redefined, self.func_ir, typ)
        elif isinstance(inst, (ir.StaticSetItem, ir.SetItem)):
            index = inst.index if isinstance(inst, ir.SetItem) else inst.index_var
            result = guard(self._index_to_shape, scope, equiv_set, inst.target, index)
            if not result:
                return ([], [])
            if result[0] is not None:
                assert isinstance(inst, (ir.StaticSetItem, ir.SetItem))
                inst.index = result[0]
            result = result[1]
            target_shape = result.kwargs['shape']
            if 'pre' in result.kwargs:
                pre = result.kwargs['pre']
            value_shape = equiv_set.get_shape(inst.value)
            if value_shape == ():
                equiv_set.set_shape_setitem(inst, target_shape)
                return (pre, [])
            elif value_shape is not None:
                target_typ = self.typemap[inst.target.name]
                require(isinstance(target_typ, types.ArrayCompatible))
                target_ndim = target_typ.ndim
                shapes = [target_shape, value_shape]
                names = [inst.target.name, inst.value.name]
                broadcast_result = self._broadcast_assert_shapes(scope, equiv_set, inst.loc, shapes, names)
                require('shape' in broadcast_result.kwargs)
                require('pre' in broadcast_result.kwargs)
                shape = broadcast_result.kwargs['shape']
                asserts = broadcast_result.kwargs['pre']
                n = len(shape)
                assert target_ndim >= n
                equiv_set.set_shape_setitem(inst, shape)
                return (pre + asserts, [])
            else:
                return (pre, [])
        elif isinstance(inst, ir.Branch):

            def handle_call_binop(cond_def):
                if False:
                    while True:
                        i = 10
                br = None
                if cond_def.fn == operator.eq:
                    br = inst.truebr
                    otherbr = inst.falsebr
                    cond_val = 1
                elif cond_def.fn == operator.ne:
                    br = inst.falsebr
                    otherbr = inst.truebr
                    cond_val = 0
                lhs_typ = self.typemap[cond_def.lhs.name]
                rhs_typ = self.typemap[cond_def.rhs.name]
                if br is not None and (isinstance(lhs_typ, types.Integer) and isinstance(rhs_typ, types.Integer) or (isinstance(lhs_typ, types.BaseTuple) and isinstance(rhs_typ, types.BaseTuple))):
                    loc = inst.loc
                    args = (cond_def.lhs, cond_def.rhs)
                    asserts = self._make_assert_equiv(scope, loc, equiv_set, args)
                    asserts.append(ir.Assign(ir.Const(cond_val, loc), cond_var, loc))
                    self.prepends[label, br] = asserts
                    self.prepends[label, otherbr] = [ir.Assign(ir.Const(1 - cond_val, loc), cond_var, loc)]
            cond_var = inst.cond
            cond_def = guard(get_definition, self.func_ir, cond_var)
            if not cond_def:
                equivs = equiv_set.get_equiv_set(cond_var)
                defs = []
                for name in equivs:
                    if isinstance(name, str) and name in self.typemap:
                        var_def = guard(get_definition, self.func_ir, name, lhs_only=True)
                        if isinstance(var_def, ir.Var):
                            var_def = var_def.name
                        if var_def:
                            defs.append(var_def)
                    else:
                        defs.append(name)
                defvars = set(filter(lambda x: isinstance(x, str), defs))
                defconsts = set(defs).difference(defvars)
                if len(defconsts) == 1:
                    cond_def = list(defconsts)[0]
                elif len(defvars) == 1:
                    cond_def = guard(get_definition, self.func_ir, list(defvars)[0])
            if isinstance(cond_def, ir.Expr) and cond_def.op == 'binop':
                handle_call_binop(cond_def)
            elif isinstance(cond_def, ir.Expr) and cond_def.op == 'call':
                glbl_bool = guard(get_definition, self.func_ir, cond_def.func)
                if glbl_bool is not None and glbl_bool.value is bool:
                    if len(cond_def.args) == 1:
                        condition = guard(get_definition, self.func_ir, cond_def.args[0])
                        if condition is not None and isinstance(condition, ir.Expr) and (condition.op == 'binop'):
                            handle_call_binop(condition)
            else:
                if isinstance(cond_def, ir.Const):
                    cond_def = cond_def.value
                if isinstance(cond_def, int) or isinstance(cond_def, bool):
                    pruned_br = inst.falsebr if cond_def else inst.truebr
                    if pruned_br in self.pruned_predecessors:
                        self.pruned_predecessors[pruned_br].append(label)
                    else:
                        self.pruned_predecessors[pruned_br] = [label]
        elif type(inst) in array_analysis_extensions:
            f = array_analysis_extensions[type(inst)]
            (pre, post) = f(inst, equiv_set, self.typemap, self)
        return (pre, post)

    def _analyze_expr(self, scope, equiv_set, expr, lhs):
        if False:
            while True:
                i = 10
        fname = '_analyze_op_{}'.format(expr.op)
        try:
            fn = getattr(self, fname)
        except AttributeError:
            return None
        return guard(fn, scope, equiv_set, expr, lhs)

    def _analyze_op_getattr(self, scope, equiv_set, expr, lhs):
        if False:
            return 10
        if expr.attr == 'T' and self._isarray(expr.value.name):
            return self._analyze_op_call_numpy_transpose(scope, equiv_set, expr.loc, [expr.value], {})
        elif expr.attr == 'shape':
            shape = equiv_set.get_shape(expr.value)
            return ArrayAnalysis.AnalyzeResult(shape=shape)
        elif expr.attr in ('real', 'imag') and self._isarray(expr.value.name):
            return ArrayAnalysis.AnalyzeResult(shape=expr.value)
        elif self._isarray(lhs.name):
            canonical_value = get_canonical_alias(expr.value.name, self.alias_map)
            if (canonical_value, expr.attr) in self.object_attrs:
                return ArrayAnalysis.AnalyzeResult(shape=self.object_attrs[canonical_value, expr.attr])
            else:
                typ = self.typemap[lhs.name]
                post = []
                shape = self._gen_shape_call(equiv_set, lhs, typ.ndim, None, post)
                self.object_attrs[canonical_value, expr.attr] = shape
                return ArrayAnalysis.AnalyzeResult(shape=shape, post=post)
        return None

    def _analyze_op_cast(self, scope, equiv_set, expr, lhs):
        if False:
            print('Hello World!')
        return ArrayAnalysis.AnalyzeResult(shape=expr.value)

    def _analyze_op_exhaust_iter(self, scope, equiv_set, expr, lhs):
        if False:
            return 10
        var = expr.value
        typ = self.typemap[var.name]
        if isinstance(typ, types.BaseTuple):
            require(len(typ) == expr.count)
            require(equiv_set.has_shape(var))
            return ArrayAnalysis.AnalyzeResult(shape=var)
        return None

    def gen_literal_slice_part(self, arg_val, loc, scope, stmts, equiv_set, name='static_literal_slice_part'):
        if False:
            i = 10
            return i + 15
        static_literal_slice_part_var = ir.Var(scope, mk_unique_var(name), loc)
        static_literal_slice_part_val = ir.Const(arg_val, loc)
        static_literal_slice_part_typ = types.IntegerLiteral(arg_val)
        stmts.append(ir.Assign(value=static_literal_slice_part_val, target=static_literal_slice_part_var, loc=loc))
        self._define(equiv_set, static_literal_slice_part_var, static_literal_slice_part_typ, static_literal_slice_part_val)
        return (static_literal_slice_part_var, static_literal_slice_part_typ)

    def gen_static_slice_size(self, lhs_rel, rhs_rel, loc, scope, stmts, equiv_set):
        if False:
            for i in range(10):
                print('nop')
        (the_var, *_) = self.gen_literal_slice_part(rhs_rel - lhs_rel, loc, scope, stmts, equiv_set, name='static_slice_size')
        return the_var

    def gen_explicit_neg(self, arg, arg_rel, arg_typ, size_typ, loc, scope, dsize, stmts, equiv_set):
        if False:
            i = 10
            return i + 15
        assert not isinstance(size_typ, int)
        explicit_neg_var = ir.Var(scope, mk_unique_var('explicit_neg'), loc)
        explicit_neg_val = ir.Expr.binop(operator.add, dsize, arg, loc=loc)
        explicit_neg_typ = types.intp
        self.calltypes[explicit_neg_val] = signature(explicit_neg_typ, size_typ, arg_typ)
        stmts.append(ir.Assign(value=explicit_neg_val, target=explicit_neg_var, loc=loc))
        self._define(equiv_set, explicit_neg_var, explicit_neg_typ, explicit_neg_val)
        return (explicit_neg_var, explicit_neg_typ)

    def update_replacement_slice(self, lhs, lhs_typ, lhs_rel, dsize_rel, replacement_slice, slice_index, need_replacement, loc, scope, stmts, equiv_set, size_typ, dsize):
        if False:
            while True:
                i = 10
        known = False
        if isinstance(lhs_rel, int):
            if lhs_rel == 0:
                known = True
            elif isinstance(dsize_rel, int):
                known = True
                wil = wrap_index_literal(lhs_rel, dsize_rel)
                if wil != lhs_rel:
                    if config.DEBUG_ARRAY_OPT >= 2:
                        print('Replacing slice to hard-code known slice size.')
                    need_replacement = True
                    (literal_var, literal_typ) = self.gen_literal_slice_part(wil, loc, scope, stmts, equiv_set)
                    assert slice_index == 0 or slice_index == 1
                    if slice_index == 0:
                        replacement_slice.args = (literal_var, replacement_slice.args[1])
                    else:
                        replacement_slice.args = (replacement_slice.args[0], literal_var)
                    lhs = replacement_slice.args[slice_index]
                    lhs_typ = literal_typ
                    lhs_rel = equiv_set.get_rel(lhs)
            elif lhs_rel < 0:
                need_replacement = True
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Replacing slice due to known negative index.')
                (explicit_neg_var, explicit_neg_typ) = self.gen_explicit_neg(lhs, lhs_rel, lhs_typ, size_typ, loc, scope, dsize, stmts, equiv_set)
                if slice_index == 0:
                    replacement_slice.args = (explicit_neg_var, replacement_slice.args[1])
                else:
                    replacement_slice.args = (replacement_slice.args[0], explicit_neg_var)
                lhs = replacement_slice.args[slice_index]
                lhs_typ = explicit_neg_typ
                lhs_rel = equiv_set.get_rel(lhs)
        return (lhs, lhs_typ, lhs_rel, replacement_slice, need_replacement, known)

    def slice_size(self, index, dsize, equiv_set, scope, stmts):
        if False:
            return 10
        'Reason about the size of a slice represented by the "index"\n        variable, and return a variable that has this size data, or\n        raise GuardException if it cannot reason about it.\n\n        The computation takes care of negative values used in the slice\n        with respect to the given dimensional size ("dsize").\n\n        Extra statements required to produce the result are appended\n        to parent function\'s stmts list.\n        '
        loc = index.loc
        index_def = get_definition(self.func_ir, index)
        (fname, mod_name) = find_callname(self.func_ir, index_def, typemap=self.typemap)
        require(fname == 'slice' and mod_name in 'builtins')
        require(len(index_def.args) == 2)
        lhs = index_def.args[0]
        rhs = index_def.args[1]
        size_typ = self.typemap[dsize.name]
        lhs_typ = self.typemap[lhs.name]
        rhs_typ = self.typemap[rhs.name]
        if config.DEBUG_ARRAY_OPT >= 2:
            print(f'slice_size index={index} dsize={dsize} index_def={index_def} lhs={lhs} rhs={rhs} size_typ={size_typ} lhs_typ={lhs_typ} rhs_typ={rhs_typ}')
        replacement_slice = copy.deepcopy(index_def)
        need_replacement = False
        if isinstance(lhs_typ, types.NoneType):
            zero_var = ir.Var(scope, mk_unique_var('zero'), loc)
            zero = ir.Const(0, loc)
            stmts.append(ir.Assign(value=zero, target=zero_var, loc=loc))
            self._define(equiv_set, zero_var, types.IntegerLiteral(0), zero)
            lhs = zero_var
            lhs_typ = types.IntegerLiteral(0)
            replacement_slice.args = (lhs, replacement_slice.args[1])
            need_replacement = True
            if config.DEBUG_ARRAY_OPT >= 2:
                print('Replacing slice because lhs is None.')
        if isinstance(rhs_typ, types.NoneType):
            rhs = dsize
            rhs_typ = size_typ
            replacement_slice.args = (replacement_slice.args[0], rhs)
            need_replacement = True
            if config.DEBUG_ARRAY_OPT >= 2:
                print('Replacing slice because lhs is None.')
        lhs_rel = equiv_set.get_rel(lhs)
        rhs_rel = equiv_set.get_rel(rhs)
        dsize_rel = equiv_set.get_rel(dsize)
        if config.DEBUG_ARRAY_OPT >= 2:
            print('lhs_rel', lhs_rel, 'rhs_rel', rhs_rel, 'dsize_rel', dsize_rel)
        [lhs, lhs_typ, lhs_rel, replacement_slice, need_replacement, lhs_known] = self.update_replacement_slice(lhs, lhs_typ, lhs_rel, dsize_rel, replacement_slice, 0, need_replacement, loc, scope, stmts, equiv_set, size_typ, dsize)
        [rhs, rhs_typ, rhs_rel, replacement_slice, need_replacement, rhs_known] = self.update_replacement_slice(rhs, rhs_typ, rhs_rel, dsize_rel, replacement_slice, 1, need_replacement, loc, scope, stmts, equiv_set, size_typ, dsize)
        if config.DEBUG_ARRAY_OPT >= 2:
            print('lhs_known:', lhs_known)
            print('rhs_known:', rhs_known)
        if not need_replacement:
            replacement_slice_var = None
        else:
            replacement_slice_var = ir.Var(scope, mk_unique_var('replacement_slice'), loc)
            new_arg_typs = (types.intp, types.intp)
            rs_calltype = self.typemap[index_def.func.name].get_call_type(self.context, new_arg_typs, {})
            self.calltypes[replacement_slice] = rs_calltype
            stmts.append(ir.Assign(value=replacement_slice, target=replacement_slice_var, loc=loc))
            self.typemap[replacement_slice_var.name] = self.typemap[index.name]
        if config.DEBUG_ARRAY_OPT >= 2:
            print('after rewriting negatives', 'lhs_rel', lhs_rel, 'rhs_rel', rhs_rel)
        if lhs_known and rhs_known:
            if config.DEBUG_ARRAY_OPT >= 2:
                print('lhs and rhs known so return static size')
            return (self.gen_static_slice_size(lhs_rel, rhs_rel, loc, scope, stmts, equiv_set), replacement_slice_var)
        if lhs_rel == 0 and isinstance(rhs_rel, tuple) and equiv_set.is_equiv(dsize, rhs_rel[0]) and (rhs_rel[1] == 0):
            return (dsize, None)
        slice_typ = types.intp
        orig_slice_typ = slice_typ
        size_var = ir.Var(scope, mk_unique_var('slice_size'), loc)
        size_val = ir.Expr.binop(operator.sub, rhs, lhs, loc=loc)
        self.calltypes[size_val] = signature(slice_typ, rhs_typ, lhs_typ)
        self._define(equiv_set, size_var, slice_typ, size_val)
        size_rel = equiv_set.get_rel(size_var)
        if config.DEBUG_ARRAY_OPT >= 2:
            print('size_rel', size_rel, type(size_rel))
        wrap_var = ir.Var(scope, mk_unique_var('wrap'), loc)
        wrap_def = ir.Global('wrap_index', wrap_index, loc=loc)
        fnty = get_global_func_typ(wrap_index)
        sig = self.context.resolve_function_type(fnty, (orig_slice_typ, size_typ), {})
        self._define(equiv_set, wrap_var, fnty, wrap_def)

        def gen_wrap_if_not_known(val, val_typ, known):
            if False:
                for i in range(10):
                    print('nop')
            if not known:
                var = ir.Var(scope, mk_unique_var('var'), loc)
                var_typ = types.intp
                new_value = ir.Expr.call(wrap_var, [val, dsize], {}, loc)
                self._define(equiv_set, var, var_typ, new_value)
                self.calltypes[new_value] = sig
                return (var, var_typ, new_value)
            else:
                return (val, val_typ, None)
        (var1, var1_typ, value1) = gen_wrap_if_not_known(lhs, lhs_typ, lhs_known)
        (var2, var2_typ, value2) = gen_wrap_if_not_known(rhs, rhs_typ, rhs_known)
        stmts.append(ir.Assign(value=size_val, target=size_var, loc=loc))
        stmts.append(ir.Assign(value=wrap_def, target=wrap_var, loc=loc))
        if value1 is not None:
            stmts.append(ir.Assign(value=value1, target=var1, loc=loc))
        if value2 is not None:
            stmts.append(ir.Assign(value=value2, target=var2, loc=loc))
        post_wrap_size_var = ir.Var(scope, mk_unique_var('post_wrap_slice_size'), loc)
        post_wrap_size_val = ir.Expr.binop(operator.sub, var2, var1, loc=loc)
        self.calltypes[post_wrap_size_val] = signature(slice_typ, var2_typ, var1_typ)
        self._define(equiv_set, post_wrap_size_var, slice_typ, post_wrap_size_val)
        stmts.append(ir.Assign(value=post_wrap_size_val, target=post_wrap_size_var, loc=loc))
        if isinstance(size_rel, tuple):
            if config.DEBUG_ARRAY_OPT >= 2:
                print('size_rel is tuple', equiv_set.rel_map)
            rel_map_entry = None
            for (rme, rme_tuple) in equiv_set.rel_map.items():
                if rme[1] == size_rel[1] and equiv_set.is_equiv(rme[0], size_rel[0]):
                    rel_map_entry = rme_tuple
                    break
            if rel_map_entry is not None:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('establishing equivalence to', rel_map_entry)
                equiv_set.insert_equiv(size_var, rel_map_entry[0])
                equiv_set.insert_equiv(post_wrap_size_var, rel_map_entry[1])
            else:
                equiv_set.rel_map[size_rel] = (size_var, post_wrap_size_var)
        return (post_wrap_size_var, replacement_slice_var)

    def _index_to_shape(self, scope, equiv_set, var, ind_var):
        if False:
            i = 10
            return i + 15
        'For indexing like var[index] (either write or read), see if\n        the index corresponds to a range/slice shape.\n        Returns a 2-tuple where the first item is either None or a ir.Var\n        to be used to replace the index variable in the outer getitem or\n        setitem instruction.  The second item is also a tuple returning\n        the shape and prepending instructions.\n        '
        typ = self.typemap[var.name]
        require(isinstance(typ, types.ArrayCompatible))
        ind_typ = self.typemap[ind_var.name]
        ind_shape = equiv_set._get_shape(ind_var)
        var_shape = equiv_set._get_shape(var)
        if isinstance(ind_typ, types.SliceType):
            seq_typs = (ind_typ,)
            seq = (ind_var,)
        else:
            require(isinstance(ind_typ, types.BaseTuple))
            (seq, op) = find_build_sequence(self.func_ir, ind_var)
            require(op == 'build_tuple')
            seq_typs = tuple((self.typemap[x.name] for x in seq))
        require(len(ind_shape) == len(seq_typs) == len(var_shape))
        stmts = []

        def to_shape(typ, index, dsize):
            if False:
                return 10
            if isinstance(typ, types.SliceType):
                return self.slice_size(index, dsize, equiv_set, scope, stmts)
            elif isinstance(typ, types.Number):
                return (None, None)
            else:
                require(False)
        shape_list = []
        index_var_list = []
        replace_index = False
        for (typ, size, dsize, orig_ind) in zip(seq_typs, ind_shape, var_shape, seq):
            (shape_part, index_var_part) = to_shape(typ, size, dsize)
            shape_list.append(shape_part)
            if index_var_part is not None:
                replace_index = True
                index_var_list.append(index_var_part)
            else:
                index_var_list.append(orig_ind)
        if replace_index:
            if len(index_var_list) > 1:
                replacement_build_tuple_var = ir.Var(scope, mk_unique_var('replacement_build_tuple'), ind_shape[0].loc)
                new_build_tuple = ir.Expr.build_tuple(index_var_list, ind_shape[0].loc)
                stmts.append(ir.Assign(value=new_build_tuple, target=replacement_build_tuple_var, loc=ind_shape[0].loc))
                self.typemap[replacement_build_tuple_var.name] = ind_typ
            else:
                replacement_build_tuple_var = index_var_list[0]
        else:
            replacement_build_tuple_var = None
        shape = tuple(shape_list)
        require(not all((x is None for x in shape)))
        shape = tuple((x for x in shape if x is not None))
        return (replacement_build_tuple_var, ArrayAnalysis.AnalyzeResult(shape=shape, pre=stmts))

    def _analyze_op_getitem(self, scope, equiv_set, expr, lhs):
        if False:
            return 10
        result = self._index_to_shape(scope, equiv_set, expr.value, expr.index)
        if result[0] is not None:
            expr.index = result[0]
        return result[1]

    def _analyze_op_static_getitem(self, scope, equiv_set, expr, lhs):
        if False:
            return 10
        var = expr.value
        typ = self.typemap[var.name]
        if not isinstance(typ, types.BaseTuple):
            result = self._index_to_shape(scope, equiv_set, expr.value, expr.index_var)
            if result[0] is not None:
                expr.index_var = result[0]
            return result[1]
        shape = equiv_set._get_shape(var)
        if isinstance(expr.index, int):
            require(expr.index < len(shape))
            return ArrayAnalysis.AnalyzeResult(shape=shape[expr.index])
        elif isinstance(expr.index, slice):
            return ArrayAnalysis.AnalyzeResult(shape=shape[expr.index])
        require(False)

    def _analyze_op_unary(self, scope, equiv_set, expr, lhs):
        if False:
            while True:
                i = 10
        require(expr.fn in UNARY_MAP_OP)
        if self._isarray(expr.value.name) or expr.fn == operator.add:
            return ArrayAnalysis.AnalyzeResult(shape=expr.value)
        return None

    def _analyze_op_binop(self, scope, equiv_set, expr, lhs):
        if False:
            print('Hello World!')
        require(expr.fn in BINARY_MAP_OP)
        return self._analyze_broadcast(scope, equiv_set, expr.loc, [expr.lhs, expr.rhs], expr.fn)

    def _analyze_op_inplace_binop(self, scope, equiv_set, expr, lhs):
        if False:
            return 10
        require(expr.fn in INPLACE_BINARY_MAP_OP)
        return self._analyze_broadcast(scope, equiv_set, expr.loc, [expr.lhs, expr.rhs], expr.fn)

    def _analyze_op_arrayexpr(self, scope, equiv_set, expr, lhs):
        if False:
            i = 10
            return i + 15
        return self._analyze_broadcast(scope, equiv_set, expr.loc, expr.list_vars(), None)

    def _analyze_op_build_tuple(self, scope, equiv_set, expr, lhs):
        if False:
            i = 10
            return i + 15
        for x in expr.items:
            if isinstance(x, ir.Var) and isinstance(self.typemap[x.name], types.ArrayCompatible) and (self.typemap[x.name].ndim > 1):
                return None
        consts = []
        for var in expr.items:
            x = guard(find_const, self.func_ir, var)
            if x is not None:
                consts.append(x)
            else:
                break
        else:
            out = tuple([ir.Const(x, expr.loc) for x in consts])
            return ArrayAnalysis.AnalyzeResult(shape=out, rhs=ir.Const(tuple(consts), expr.loc))
        return ArrayAnalysis.AnalyzeResult(shape=tuple(expr.items))

    def _analyze_op_call(self, scope, equiv_set, expr, lhs):
        if False:
            return 10
        from numba.stencils.stencil import StencilFunc
        callee = expr.func
        callee_def = get_definition(self.func_ir, callee)
        if isinstance(callee_def, (ir.Global, ir.FreeVar)) and is_namedtuple_class(callee_def.value):
            return ArrayAnalysis.AnalyzeResult(shape=tuple(expr.args))
        if isinstance(callee_def, (ir.Global, ir.FreeVar)) and isinstance(callee_def.value, StencilFunc):
            args = expr.args
            return self._analyze_stencil(scope, equiv_set, callee_def.value, expr.loc, args, dict(expr.kws))
        (fname, mod_name) = find_callname(self.func_ir, expr, typemap=self.typemap)
        added_mod_name = False
        if isinstance(mod_name, ir.Var) and isinstance(self.typemap[mod_name.name], types.ArrayCompatible):
            args = [mod_name] + expr.args
            mod_name = 'numpy'
            added_mod_name = True
        else:
            args = expr.args
        fname = '_analyze_op_call_{}_{}'.format(mod_name, fname).replace('.', '_')
        if fname in UFUNC_MAP_OP:
            return self._analyze_broadcast(scope, equiv_set, expr.loc, args, None)
        else:
            try:
                fn = getattr(self, fname)
            except AttributeError:
                return None
            result = guard(fn, scope=scope, equiv_set=equiv_set, loc=expr.loc, args=args, kws=dict(expr.kws))
            if added_mod_name:
                expr.args = args[1:]
            return result

    def _analyze_op_call_builtins_len(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        require(len(args) == 1)
        var = args[0]
        typ = self.typemap[var.name]
        require(isinstance(typ, types.ArrayCompatible))
        shape = equiv_set._get_shape(var)
        return ArrayAnalysis.AnalyzeResult(shape=shape[0], rhs=shape[0])

    def _analyze_op_call_numba_parfors_array_analysis_assert_equiv(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        equiv_set.insert_equiv(*args[1:])
        return None

    def _analyze_op_call_numba_parfors_array_analysis_wrap_index(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        ' Analyze wrap_index calls added by a previous run of\n            Array Analysis\n        '
        require(len(args) == 2)
        slice_size = args[0].name
        dim_size = args[1].name
        slice_eq = equiv_set._get_or_add_ind(slice_size)
        dim_eq = equiv_set._get_or_add_ind(dim_size)
        if (slice_eq, dim_eq) in equiv_set.wrap_map:
            wrap_ind = equiv_set.wrap_map[slice_eq, dim_eq]
            require(wrap_ind in equiv_set.ind_to_var)
            vs = equiv_set.ind_to_var[wrap_ind]
            require(vs != [])
            return ArrayAnalysis.AnalyzeResult(shape=(vs[0],))
        else:
            return ArrayAnalysis.AnalyzeResult(shape=WrapIndexMeta(slice_eq, dim_eq))

    def _analyze_numpy_create_array(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        shape_var = None
        if len(args) > 0:
            shape_var = args[0]
        elif 'shape' in kws:
            shape_var = kws['shape']
        if shape_var:
            return ArrayAnalysis.AnalyzeResult(shape=shape_var)
        raise errors.UnsupportedRewriteError('Must specify a shape for array creation', loc=loc)

    def _analyze_op_call_numpy_empty(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return self._analyze_numpy_create_array(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numba_np_unsafe_ndarray_empty_inferred(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return self._analyze_numpy_create_array(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numpy_zeros(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_numpy_create_array(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numpy_ones(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        return self._analyze_numpy_create_array(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numpy_eye(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        if len(args) > 0:
            N = args[0]
        elif 'N' in kws:
            N = kws['N']
        else:
            raise errors.UnsupportedRewriteError("Expect one argument (or 'N') to eye function", loc=loc)
        if 'M' in kws:
            M = kws['M']
        else:
            M = N
        return ArrayAnalysis.AnalyzeResult(shape=(N, M))

    def _analyze_op_call_numpy_identity(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        assert len(args) > 0
        N = args[0]
        return ArrayAnalysis.AnalyzeResult(shape=(N, N))

    def _analyze_op_call_numpy_diag(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        assert len(args) > 0
        a = args[0]
        assert isinstance(a, ir.Var)
        atyp = self.typemap[a.name]
        if isinstance(atyp, types.ArrayCompatible):
            if atyp.ndim == 2:
                if 'k' in kws:
                    k = kws['k']
                    if not equiv_set.is_equiv(k, 0):
                        return None
                (m, n) = equiv_set._get_shape(a)
                if equiv_set.is_equiv(m, n):
                    return ArrayAnalysis.AnalyzeResult(shape=(m,))
            elif atyp.ndim == 1:
                (m,) = equiv_set._get_shape(a)
                return ArrayAnalysis.AnalyzeResult(shape=(m, m))
        return None

    def _analyze_numpy_array_like(self, scope, equiv_set, args, kws):
        if False:
            return 10
        assert len(args) > 0
        var = args[0]
        typ = self.typemap[var.name]
        if isinstance(typ, types.Integer):
            return ArrayAnalysis.AnalyzeResult(shape=(1,))
        elif isinstance(typ, types.ArrayCompatible) and equiv_set.has_shape(var):
            return ArrayAnalysis.AnalyzeResult(shape=var)
        return None

    def _analyze_op_call_numpy_ravel(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        assert len(args) == 1
        var = args[0]
        typ = self.typemap[var.name]
        assert isinstance(typ, types.ArrayCompatible)
        if typ.ndim == 1 and equiv_set.has_shape(var):
            if typ.layout == 'C':
                return ArrayAnalysis.AnalyzeResult(shape=var, rhs=var)
            else:
                return ArrayAnalysis.AnalyzeResult(shape=var)
        return None

    def _analyze_op_call_numpy_copy(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_numpy_array_like(scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_empty_like(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_numpy_array_like(scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_zeros_like(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_numpy_array_like(scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_ones_like(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        return self._analyze_numpy_array_like(scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_full_like(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return self._analyze_numpy_array_like(scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_asfortranarray(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_numpy_array_like(scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_reshape(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        n = len(args)
        assert n > 1
        if n == 2:
            typ = self.typemap[args[1].name]
            if isinstance(typ, types.BaseTuple):
                return ArrayAnalysis.AnalyzeResult(shape=args[1])
        stmts = []
        neg_one_index = -1
        for arg_index in range(1, len(args)):
            reshape_arg = args[arg_index]
            reshape_arg_def = guard(get_definition, self.func_ir, reshape_arg)
            if isinstance(reshape_arg_def, ir.Const):
                if reshape_arg_def.value < 0:
                    if neg_one_index == -1:
                        neg_one_index = arg_index
                    else:
                        msg = 'The reshape API may only include one negative argument.'
                        raise errors.UnsupportedRewriteError(msg, loc=reshape_arg.loc)
        if neg_one_index >= 0:
            loc = args[0].loc
            calc_size_var = ir.Var(scope, mk_unique_var('calc_size_var'), loc)
            self.typemap[calc_size_var.name] = types.intp
            init_calc_var = ir.Assign(ir.Expr.getattr(args[0], 'size', loc), calc_size_var, loc)
            stmts.append(init_calc_var)
            for arg_index in range(1, len(args)):
                if arg_index == neg_one_index:
                    continue
                div_calc_size_var = ir.Var(scope, mk_unique_var('calc_size_var'), loc)
                self.typemap[div_calc_size_var.name] = types.intp
                new_binop = ir.Expr.binop(operator.floordiv, calc_size_var, args[arg_index], loc)
                div_calc = ir.Assign(new_binop, div_calc_size_var, loc)
                self.calltypes[new_binop] = signature(types.intp, types.intp, types.intp)
                stmts.append(div_calc)
                calc_size_var = div_calc_size_var
            args[neg_one_index] = calc_size_var
        return ArrayAnalysis.AnalyzeResult(shape=tuple(args[1:]), pre=stmts)

    def _analyze_op_call_numpy_transpose(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        in_arr = args[0]
        typ = self.typemap[in_arr.name]
        assert isinstance(typ, types.ArrayCompatible), 'Invalid np.transpose argument'
        shape = equiv_set._get_shape(in_arr)
        if len(args) == 1:
            return ArrayAnalysis.AnalyzeResult(shape=tuple(reversed(shape)))
        axes = [guard(find_const, self.func_ir, a) for a in args[1:]]
        if isinstance(axes[0], tuple):
            axes = list(axes[0])
        if None in axes:
            return None
        ret = [shape[i] for i in axes]
        return ArrayAnalysis.AnalyzeResult(shape=tuple(ret))

    def _analyze_op_call_numpy_random_rand(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        if len(args) > 0:
            return ArrayAnalysis.AnalyzeResult(shape=tuple(args))
        return None

    def _analyze_op_call_numpy_random_randn(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_op_call_numpy_random_rand(scope, equiv_set, loc, args, kws)

    def _analyze_op_numpy_random_with_size(self, pos, scope, equiv_set, args, kws):
        if False:
            while True:
                i = 10
        if 'size' in kws:
            return ArrayAnalysis.AnalyzeResult(shape=kws['size'])
        if len(args) > pos:
            return ArrayAnalysis.AnalyzeResult(shape=args[pos])
        return None

    def _analyze_op_call_numpy_random_ranf(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return self._analyze_op_numpy_random_with_size(0, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_random_sample(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_op_numpy_random_with_size(0, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_sample(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_op_numpy_random_with_size(0, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_random(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_op_numpy_random_with_size(0, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_standard_normal(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_op_numpy_random_with_size(0, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_chisquare(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_weibull(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_power(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_geometric(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_exponential(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_poisson(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_rayleigh(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_op_numpy_random_with_size(1, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_normal(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_uniform(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_beta(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_binomial(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_f(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_gamma(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_lognormal(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_laplace(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_randint(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        return self._analyze_op_numpy_random_with_size(2, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_random_triangular(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        return self._analyze_op_numpy_random_with_size(3, scope, equiv_set, args, kws)

    def _analyze_op_call_numpy_concatenate(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        assert len(args) > 0
        loc = args[0].loc
        (seq, op) = find_build_sequence(self.func_ir, args[0])
        n = len(seq)
        require(n > 0)
        axis = 0
        if 'axis' in kws:
            if isinstance(kws['axis'], int):
                axis = kws['axis']
            else:
                axis = find_const(self.func_ir, kws['axis'])
        elif len(args) > 1:
            axis = find_const(self.func_ir, args[1])
        require(isinstance(axis, int))
        require(op == 'build_tuple')
        shapes = [equiv_set._get_shape(x) for x in seq]
        if axis < 0:
            axis = len(shapes[0]) + axis
        require(0 <= axis < len(shapes[0]))
        asserts = []
        new_shape = []
        if n == 1:
            shape = shapes[0]
            n = equiv_set.get_equiv_const(shapes[0])
            shape.pop(0)
            for i in range(len(shape)):
                if i == axis:
                    m = equiv_set.get_equiv_const(shape[i])
                    size = m * n if m and n else None
                else:
                    size = self._sum_size(equiv_set, shapes[0])
            new_shape.append(size)
        else:
            for i in range(len(shapes[0])):
                if i == axis:
                    size = self._sum_size(equiv_set, [shape[i] for shape in shapes])
                else:
                    sizes = [shape[i] for shape in shapes]
                    asserts.append(self._call_assert_equiv(scope, loc, equiv_set, sizes))
                    size = sizes[0]
                new_shape.append(size)
        return ArrayAnalysis.AnalyzeResult(shape=tuple(new_shape), pre=sum(asserts, []))

    def _analyze_op_call_numpy_stack(self, scope, equiv_set, loc, args, kws):
        if False:
            i = 10
            return i + 15
        assert len(args) > 0
        loc = args[0].loc
        (seq, op) = find_build_sequence(self.func_ir, args[0])
        n = len(seq)
        require(n > 0)
        axis = 0
        if 'axis' in kws:
            if isinstance(kws['axis'], int):
                axis = kws['axis']
            else:
                axis = find_const(self.func_ir, kws['axis'])
        elif len(args) > 1:
            axis = find_const(self.func_ir, args[1])
        require(isinstance(axis, int))
        require(op == 'build_tuple')
        shapes = [equiv_set._get_shape(x) for x in seq]
        asserts = self._call_assert_equiv(scope, loc, equiv_set, seq)
        shape = shapes[0]
        if axis < 0:
            axis = len(shape) + axis + 1
        require(0 <= axis <= len(shape))
        new_shape = list(shape[0:axis]) + [n] + list(shape[axis:])
        return ArrayAnalysis.AnalyzeResult(shape=tuple(new_shape), pre=asserts)

    def _analyze_op_call_numpy_vstack(self, scope, equiv_set, loc, args, kws):
        if False:
            for i in range(10):
                print('nop')
        assert len(args) == 1
        (seq, op) = find_build_sequence(self.func_ir, args[0])
        n = len(seq)
        require(n > 0)
        typ = self.typemap[seq[0].name]
        require(isinstance(typ, types.ArrayCompatible))
        if typ.ndim < 2:
            return self._analyze_op_call_numpy_stack(scope, equiv_set, loc, args, kws)
        else:
            kws['axis'] = 0
            return self._analyze_op_call_numpy_concatenate(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numpy_hstack(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        assert len(args) == 1
        (seq, op) = find_build_sequence(self.func_ir, args[0])
        n = len(seq)
        require(n > 0)
        typ = self.typemap[seq[0].name]
        require(isinstance(typ, types.ArrayCompatible))
        if typ.ndim < 2:
            kws['axis'] = 0
        else:
            kws['axis'] = 1
        return self._analyze_op_call_numpy_concatenate(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numpy_dstack(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        assert len(args) == 1
        (seq, op) = find_build_sequence(self.func_ir, args[0])
        n = len(seq)
        require(n > 0)
        typ = self.typemap[seq[0].name]
        require(isinstance(typ, types.ArrayCompatible))
        if typ.ndim == 1:
            kws['axis'] = 1
            result = self._analyze_op_call_numpy_stack(scope, equiv_set, loc, args, kws)
            require(result)
            result.kwargs['shape'] = tuple([1] + list(result.kwargs['shape']))
            return result
        elif typ.ndim == 2:
            kws['axis'] = 2
            return self._analyze_op_call_numpy_stack(scope, equiv_set, loc, args, kws)
        else:
            kws['axis'] = 2
            return self._analyze_op_call_numpy_concatenate(scope, equiv_set, loc, args, kws)

    def _analyze_op_call_numpy_cumsum(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        return None

    def _analyze_op_call_numpy_cumprod(self, scope, equiv_set, loc, args, kws):
        if False:
            return 10
        return None

    def _analyze_op_call_numpy_linspace(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        n = len(args)
        num = 50
        if n > 2:
            num = args[2]
        elif 'num' in kws:
            num = kws['num']
        return ArrayAnalysis.AnalyzeResult(shape=(num,))

    def _analyze_op_call_numpy_dot(self, scope, equiv_set, loc, args, kws):
        if False:
            while True:
                i = 10
        n = len(args)
        assert n >= 2
        loc = args[0].loc
        require(all([self._isarray(x.name) for x in args]))
        typs = [self.typemap[x.name] for x in args]
        dims = [ty.ndim for ty in typs]
        require(all((x > 0 for x in dims)))
        if dims[0] == 1 and dims[1] == 1:
            return None
        shapes = [equiv_set._get_shape(x) for x in args]
        if dims[0] == 1:
            asserts = self._call_assert_equiv(scope, loc, equiv_set, [shapes[0][0], shapes[1][-2]])
            return ArrayAnalysis.AnalyzeResult(shape=tuple(shapes[1][0:-2] + shapes[1][-1:]), pre=asserts)
        if dims[1] == 1:
            asserts = self._call_assert_equiv(scope, loc, equiv_set, [shapes[0][-1], shapes[1][0]])
            return ArrayAnalysis.AnalyzeResult(shape=tuple(shapes[0][0:-1]), pre=asserts)
        if dims[0] == 2 and dims[1] == 2:
            asserts = self._call_assert_equiv(scope, loc, equiv_set, [shapes[0][1], shapes[1][0]])
            return ArrayAnalysis.AnalyzeResult(shape=(shapes[0][0], shapes[1][1]), pre=asserts)
        if dims[0] > 2:
            pass
        return None

    def _analyze_stencil(self, scope, equiv_set, stencil_func, loc, args, kws):
        if False:
            i = 10
            return i + 15
        std_idx_arrs = stencil_func.options.get('standard_indexing', ())
        kernel_arg_names = stencil_func.kernel_ir.arg_names
        if isinstance(std_idx_arrs, str):
            std_idx_arrs = (std_idx_arrs,)
        rel_idx_arrs = []
        assert len(args) > 0 and len(args) == len(kernel_arg_names)
        for (arg, var) in zip(kernel_arg_names, args):
            typ = self.typemap[var.name]
            if isinstance(typ, types.ArrayCompatible) and (not arg in std_idx_arrs):
                rel_idx_arrs.append(var)
        n = len(rel_idx_arrs)
        require(n > 0)
        asserts = self._call_assert_equiv(scope, loc, equiv_set, rel_idx_arrs)
        shape = equiv_set.get_shape(rel_idx_arrs[0])
        return ArrayAnalysis.AnalyzeResult(shape=shape, pre=asserts)

    def _analyze_op_call_numpy_linalg_inv(self, scope, equiv_set, loc, args, kws):
        if False:
            print('Hello World!')
        require(len(args) >= 1)
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set._get_shape(args[0]))

    def _analyze_broadcast(self, scope, equiv_set, loc, args, fn):
        if False:
            for i in range(10):
                print('nop')
        'Infer shape equivalence of arguments based on Numpy broadcast rules\n        and return shape of output\n        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html\n        '
        tups = list(filter(lambda a: self._istuple(a.name), args))
        if len(tups) == 2 and fn.__name__ == 'add':
            tup0typ = self.typemap[tups[0].name]
            tup1typ = self.typemap[tups[1].name]
            if tup0typ.count == 0:
                return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(tups[1]))
            if tup1typ.count == 0:
                return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(tups[0]))
            try:
                shapes = [equiv_set.get_shape(x) for x in tups]
                if None in shapes:
                    return None
                concat_shapes = sum(shapes, ())
                return ArrayAnalysis.AnalyzeResult(shape=concat_shapes)
            except GuardException:
                return None
        arrs = list(filter(lambda a: self._isarray(a.name), args))
        require(len(arrs) > 0)
        names = [x.name for x in arrs]
        dims = [self.typemap[x.name].ndim for x in arrs]
        max_dim = max(dims)
        require(max_dim > 0)
        try:
            shapes = [equiv_set.get_shape(x) for x in arrs]
        except GuardException:
            return ArrayAnalysis.AnalyzeResult(shape=arrs[0], pre=self._call_assert_equiv(scope, loc, equiv_set, arrs))
        pre = []
        if None in shapes:
            new_shapes = []
            for (i, s) in enumerate(shapes):
                if s is None:
                    var = arrs[i]
                    typ = self.typemap[var.name]
                    shape = self._gen_shape_call(equiv_set, var, typ.ndim, None, pre)
                    new_shapes.append(shape)
                else:
                    new_shapes.append(s)
            shapes = new_shapes
        result = self._broadcast_assert_shapes(scope, equiv_set, loc, shapes, names)
        if pre:
            if 'pre' in result.kwargs:
                prev_pre = result.kwargs['pre']
            else:
                prev_pre = []
            result.kwargs['pre'] = pre + prev_pre
        return result

    def _broadcast_assert_shapes(self, scope, equiv_set, loc, shapes, names):
        if False:
            return 10
        'Produce assert_equiv for sizes in each dimension, taking into\n        account of dimension coercion and constant size of 1.\n        '
        asserts = []
        new_shape = []
        max_dim = max([len(shape) for shape in shapes])
        const_size_one = None
        for i in range(max_dim):
            sizes = []
            size_names = []
            for (name, shape) in zip(names, shapes):
                if i < len(shape):
                    size = shape[len(shape) - 1 - i]
                    const_size = equiv_set.get_equiv_const(size)
                    if const_size == 1:
                        const_size_one = size
                    else:
                        sizes.append(size)
                        size_names.append(name)
            if sizes == []:
                assert const_size_one is not None
                sizes.append(const_size_one)
                size_names.append('1')
            asserts.append(self._call_assert_equiv(scope, loc, equiv_set, sizes, names=size_names))
            new_shape.append(sizes[0])
        return ArrayAnalysis.AnalyzeResult(shape=tuple(reversed(new_shape)), pre=sum(asserts, []))

    def _call_assert_equiv(self, scope, loc, equiv_set, args, names=None):
        if False:
            i = 10
            return i + 15
        insts = self._make_assert_equiv(scope, loc, equiv_set, args, names=names)
        if len(args) > 1:
            equiv_set.insert_equiv(*args)
        return insts

    def _make_assert_equiv(self, scope, loc, equiv_set, _args, names=None):
        if False:
            i = 10
            return i + 15
        if config.DEBUG_ARRAY_OPT >= 2:
            print('make_assert_equiv:', _args, names)
        if names is None:
            names = [x.name for x in _args]
        args = []
        arg_names = []
        for (name, x) in zip(names, _args):
            if config.DEBUG_ARRAY_OPT >= 2:
                print('name, x:', name, x)
            seen = False
            for y in args:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('is equiv to?', y, equiv_set.is_equiv(x, y))
                if equiv_set.is_equiv(x, y):
                    seen = True
                    break
            if not seen:
                args.append(x)
                arg_names.append(name)
        if len(args) < 2:
            if config.DEBUG_ARRAY_OPT >= 2:
                print('Will not insert assert_equiv as args are known to be equivalent.')
            return []
        msg = 'Sizes of {} do not match on {}'.format(', '.join(arg_names), loc)
        msg_val = ir.Const(msg, loc)
        msg_typ = types.StringLiteral(msg)
        msg_var = ir.Var(scope, mk_unique_var('msg'), loc)
        self.typemap[msg_var.name] = msg_typ
        argtyps = tuple([msg_typ] + [self.typemap[x.name] for x in args])
        tup_typ = types.StarArgTuple.from_types(argtyps)
        assert_var = ir.Var(scope, mk_unique_var('assert'), loc)
        assert_def = ir.Global('assert_equiv', assert_equiv, loc=loc)
        fnty = get_global_func_typ(assert_equiv)
        sig = self.context.resolve_function_type(fnty, (tup_typ,), {})
        self._define(equiv_set, assert_var, fnty, assert_def)
        var = ir.Var(scope, mk_unique_var('ret'), loc)
        value = ir.Expr.call(assert_var, [msg_var] + args, {}, loc=loc)
        self._define(equiv_set, var, types.none, value)
        self.calltypes[value] = sig
        return [ir.Assign(value=msg_val, target=msg_var, loc=loc), ir.Assign(value=assert_def, target=assert_var, loc=loc), ir.Assign(value=value, target=var, loc=loc)]

    def _gen_shape_call(self, equiv_set, var, ndims, shape, post):
        if False:
            print('Hello World!')
        if isinstance(shape, ir.Var):
            shape = equiv_set.get_shape(shape)
        if isinstance(shape, ir.Var):
            attr_var = shape
            shape_attr_call = None
            shape = None
        elif isinstance(shape, ir.Arg):
            attr_var = var
            shape_attr_call = None
            shape = None
        else:
            shape_attr_call = ir.Expr.getattr(var, 'shape', var.loc)
            attr_var = ir.Var(var.scope, mk_unique_var('{}_shape'.format(var.name)), var.loc)
            shape_attr_typ = types.containers.UniTuple(types.intp, ndims)
        size_vars = []
        use_attr_var = False
        if shape:
            nshapes = len(shape)
            if ndims < nshapes:
                shape = shape[nshapes - ndims:]
        for i in range(ndims):
            skip = False
            if shape and shape[i]:
                if isinstance(shape[i], ir.Var):
                    typ = self.typemap[shape[i].name]
                    if isinstance(typ, (types.Number, types.SliceType)):
                        size_var = shape[i]
                        skip = True
                else:
                    if isinstance(shape[i], int):
                        size_val = ir.Const(shape[i], var.loc)
                    else:
                        size_val = shape[i]
                    assert isinstance(size_val, ir.Const)
                    size_var = ir.Var(var.scope, mk_unique_var('{}_size{}'.format(var.name, i)), var.loc)
                    post.append(ir.Assign(size_val, size_var, var.loc))
                    self._define(equiv_set, size_var, types.intp, size_val)
                    skip = True
            if not skip:
                size_var = ir.Var(var.scope, mk_unique_var('{}_size{}'.format(var.name, i)), var.loc)
                getitem = ir.Expr.static_getitem(attr_var, i, None, var.loc)
                use_attr_var = True
                self.calltypes[getitem] = None
                post.append(ir.Assign(getitem, size_var, var.loc))
                self._define(equiv_set, size_var, types.intp, getitem)
            size_vars.append(size_var)
        if use_attr_var and shape_attr_call:
            post.insert(0, ir.Assign(shape_attr_call, attr_var, var.loc))
            self._define(equiv_set, attr_var, shape_attr_typ, shape_attr_call)
        return tuple(size_vars)

    def _isarray(self, varname):
        if False:
            print('Hello World!')
        typ = self.typemap[varname]
        return isinstance(typ, types.npytypes.Array) and typ.ndim > 0

    def _istuple(self, varname):
        if False:
            for i in range(10):
                print('nop')
        typ = self.typemap[varname]
        return isinstance(typ, types.BaseTuple)

    def _sum_size(self, equiv_set, sizes):
        if False:
            return 10
        'Return the sum of the given list of sizes if they are all equivalent\n        to some constant, or None otherwise.\n        '
        s = 0
        for size in sizes:
            n = equiv_set.get_equiv_const(size)
            if n is None:
                return None
            else:
                s += n
        return s
UNARY_MAP_OP = list(npydecl.NumpyRulesUnaryArrayOperator._op_map.keys()) + [operator.pos]
BINARY_MAP_OP = npydecl.NumpyRulesArrayOperator._op_map.keys()
INPLACE_BINARY_MAP_OP = npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
UFUNC_MAP_OP = [f.__name__ for f in npydecl.supported_ufuncs]