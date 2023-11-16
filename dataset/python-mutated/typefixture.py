"""Fixture used in type-related test cases.

It contains class TypeInfos and Type objects.
"""
from __future__ import annotations
from mypy.nodes import ARG_OPT, ARG_POS, ARG_STAR, COVARIANT, MDEF, Block, ClassDef, FuncDef, SymbolTable, SymbolTableNode, TypeAlias, TypeInfo
from mypy.semanal_shared import set_callable_name
from mypy.types import AnyType, CallableType, Instance, LiteralType, NoneType, Type, TypeAliasType, TypeOfAny, TypeType, TypeVarLikeType, TypeVarTupleType, TypeVarType, UninhabitedType, UnionType

class TypeFixture:
    """Helper class that is used as a fixture in type-related unit tests.

    The members are initialized to contain various type-related values.
    """

    def __init__(self, variance: int=COVARIANT) -> None:
        if False:
            print('Hello World!')
        self.oi = self.make_type_info('builtins.object')
        self.o = Instance(self.oi, [])

        def make_type_var(name: str, id: int, values: list[Type], upper_bound: Type, variance: int) -> TypeVarType:
            if False:
                while True:
                    i = 10
            return TypeVarType(name, name, id, values, upper_bound, AnyType(TypeOfAny.from_omitted_generics), variance)
        self.t = make_type_var('T', 1, [], self.o, variance)
        self.tf = make_type_var('T', -1, [], self.o, variance)
        self.tf2 = make_type_var('T', -2, [], self.o, variance)
        self.s = make_type_var('S', 2, [], self.o, variance)
        self.s1 = make_type_var('S', 1, [], self.o, variance)
        self.sf = make_type_var('S', -2, [], self.o, variance)
        self.sf1 = make_type_var('S', -1, [], self.o, variance)
        self.u = make_type_var('U', 3, [], self.o, variance)
        self.anyt = AnyType(TypeOfAny.special_form)
        self.nonet = NoneType()
        self.uninhabited = UninhabitedType()
        self.fi = self.make_type_info('F', is_abstract=True)
        self.f2i = self.make_type_info('F2', is_abstract=True)
        self.f3i = self.make_type_info('F3', is_abstract=True, mro=[self.fi])
        self.std_tuplei = self.make_type_info('builtins.tuple', mro=[self.oi], typevars=['T'], variances=[COVARIANT])
        self.type_typei = self.make_type_info('builtins.type')
        self.bool_type_info = self.make_type_info('builtins.bool')
        self.str_type_info = self.make_type_info('builtins.str')
        self.functioni = self.make_type_info('builtins.function')
        self.ai = self.make_type_info('A', mro=[self.oi])
        self.bi = self.make_type_info('B', mro=[self.ai, self.oi])
        self.ci = self.make_type_info('C', mro=[self.ai, self.oi])
        self.di = self.make_type_info('D', mro=[self.oi])
        self.ei = self.make_type_info('E', mro=[self.fi, self.oi])
        self.e2i = self.make_type_info('E2', mro=[self.f2i, self.fi, self.oi])
        self.e3i = self.make_type_info('E3', mro=[self.fi, self.f2i, self.oi])
        self.gi = self.make_type_info('G', mro=[self.oi], typevars=['T'], variances=[variance])
        self.g2i = self.make_type_info('G2', mro=[self.oi], typevars=['T'], variances=[variance])
        self.hi = self.make_type_info('H', mro=[self.oi], typevars=['S', 'T'], variances=[variance, variance])
        self.gsi = self.make_type_info('GS', mro=[self.gi, self.oi], typevars=['T', 'S'], variances=[variance, variance], bases=[Instance(self.gi, [self.s])])
        self.gs2i = self.make_type_info('GS2', mro=[self.gi, self.oi], typevars=['S'], variances=[variance], bases=[Instance(self.gi, [self.s1])])
        self.std_listi = self.make_type_info('builtins.list', mro=[self.oi], typevars=['T'], variances=[variance])
        self.std_tuple = Instance(self.std_tuplei, [self.anyt])
        self.type_type = Instance(self.type_typei, [])
        self.function = Instance(self.functioni, [])
        self.str_type = Instance(self.str_type_info, [])
        self.bool_type = Instance(self.bool_type_info, [])
        self.a = Instance(self.ai, [])
        self.b = Instance(self.bi, [])
        self.c = Instance(self.ci, [])
        self.d = Instance(self.di, [])
        self.e = Instance(self.ei, [])
        self.e2 = Instance(self.e2i, [])
        self.e3 = Instance(self.e3i, [])
        self.f = Instance(self.fi, [])
        self.f2 = Instance(self.f2i, [])
        self.f3 = Instance(self.f3i, [])
        self.ga = Instance(self.gi, [self.a])
        self.gb = Instance(self.gi, [self.b])
        self.gd = Instance(self.gi, [self.d])
        self.go = Instance(self.gi, [self.o])
        self.gt = Instance(self.gi, [self.t])
        self.gtf = Instance(self.gi, [self.tf])
        self.gtf2 = Instance(self.gi, [self.tf2])
        self.gs = Instance(self.gi, [self.s])
        self.gdyn = Instance(self.gi, [self.anyt])
        self.gn = Instance(self.gi, [NoneType()])
        self.g2a = Instance(self.g2i, [self.a])
        self.gsaa = Instance(self.gsi, [self.a, self.a])
        self.gsab = Instance(self.gsi, [self.a, self.b])
        self.gsba = Instance(self.gsi, [self.b, self.a])
        self.gs2a = Instance(self.gs2i, [self.a])
        self.gs2b = Instance(self.gs2i, [self.b])
        self.gs2d = Instance(self.gs2i, [self.d])
        self.hab = Instance(self.hi, [self.a, self.b])
        self.haa = Instance(self.hi, [self.a, self.a])
        self.hbb = Instance(self.hi, [self.b, self.b])
        self.hts = Instance(self.hi, [self.t, self.s])
        self.had = Instance(self.hi, [self.a, self.d])
        self.hao = Instance(self.hi, [self.a, self.o])
        self.lsta = Instance(self.std_listi, [self.a])
        self.lstb = Instance(self.std_listi, [self.b])
        self.lit1 = LiteralType(1, self.a)
        self.lit2 = LiteralType(2, self.a)
        self.lit3 = LiteralType('foo', self.d)
        self.lit4 = LiteralType(4, self.a)
        self.lit1_inst = Instance(self.ai, [], last_known_value=self.lit1)
        self.lit2_inst = Instance(self.ai, [], last_known_value=self.lit2)
        self.lit3_inst = Instance(self.di, [], last_known_value=self.lit3)
        self.lit4_inst = Instance(self.ai, [], last_known_value=self.lit4)
        self.lit_str1 = LiteralType('x', self.str_type)
        self.lit_str2 = LiteralType('y', self.str_type)
        self.lit_str3 = LiteralType('z', self.str_type)
        self.lit_str1_inst = Instance(self.str_type_info, [], last_known_value=self.lit_str1)
        self.lit_str2_inst = Instance(self.str_type_info, [], last_known_value=self.lit_str2)
        self.lit_str3_inst = Instance(self.str_type_info, [], last_known_value=self.lit_str3)
        self.lit_false = LiteralType(False, self.bool_type)
        self.lit_true = LiteralType(True, self.bool_type)
        self.type_a = TypeType.make_normalized(self.a)
        self.type_b = TypeType.make_normalized(self.b)
        self.type_c = TypeType.make_normalized(self.c)
        self.type_d = TypeType.make_normalized(self.d)
        self.type_t = TypeType.make_normalized(self.t)
        self.type_any = TypeType.make_normalized(self.anyt)
        self._add_bool_dunder(self.bool_type_info)
        self._add_bool_dunder(self.ai)
        self.ub = make_type_var('UB', 5, [], self.b, variance)
        self.uc = make_type_var('UC', 6, [], self.c, variance)

        def make_type_var_tuple(name: str, id: int, upper_bound: Type) -> TypeVarTupleType:
            if False:
                while True:
                    i = 10
            return TypeVarTupleType(name, name, id, upper_bound, self.std_tuple, AnyType(TypeOfAny.from_omitted_generics))
        obj_tuple = self.std_tuple.copy_modified(args=[self.o])
        self.ts = make_type_var_tuple('Ts', 1, obj_tuple)
        self.ss = make_type_var_tuple('Ss', 2, obj_tuple)
        self.us = make_type_var_tuple('Us', 3, obj_tuple)
        self.gvi = self.make_type_info('GV', mro=[self.oi], typevars=['Ts'], typevar_tuple_index=0)
        self.gv2i = self.make_type_info('GV2', mro=[self.oi], typevars=['T', 'Ts', 'S'], typevar_tuple_index=1)

    def _add_bool_dunder(self, type_info: TypeInfo) -> None:
        if False:
            return 10
        signature = CallableType([], [], [], Instance(self.bool_type_info, []), self.function)
        bool_func = FuncDef('__bool__', [], Block([]))
        bool_func.type = set_callable_name(signature, bool_func)
        type_info.names[bool_func.name] = SymbolTableNode(MDEF, bool_func)

    def callable(self, *a: Type) -> CallableType:
        if False:
            while True:
                i = 10
        'callable(a1, ..., an, r) constructs a callable with argument types\n        a1, ... an and return type r.\n        '
        return CallableType(list(a[:-1]), [ARG_POS] * (len(a) - 1), [None] * (len(a) - 1), a[-1], self.function)

    def callable_type(self, *a: Type) -> CallableType:
        if False:
            i = 10
            return i + 15
        'callable_type(a1, ..., an, r) constructs a callable with\n        argument types a1, ... an and return type r, and which\n        represents a type.\n        '
        return CallableType(list(a[:-1]), [ARG_POS] * (len(a) - 1), [None] * (len(a) - 1), a[-1], self.type_type)

    def callable_default(self, min_args: int, *a: Type) -> CallableType:
        if False:
            for i in range(10):
                print('nop')
        'callable_default(min_args, a1, ..., an, r) constructs a\n        callable with argument types a1, ... an and return type r,\n        with min_args mandatory fixed arguments.\n        '
        n = len(a) - 1
        return CallableType(list(a[:-1]), [ARG_POS] * min_args + [ARG_OPT] * (n - min_args), [None] * n, a[-1], self.function)

    def callable_var_arg(self, min_args: int, *a: Type) -> CallableType:
        if False:
            i = 10
            return i + 15
        'callable_var_arg(min_args, a1, ..., an, r) constructs a callable\n        with argument types a1, ... *an and return type r.\n        '
        n = len(a) - 1
        return CallableType(list(a[:-1]), [ARG_POS] * min_args + [ARG_OPT] * (n - 1 - min_args) + [ARG_STAR], [None] * n, a[-1], self.function)

    def make_type_info(self, name: str, module_name: str | None=None, is_abstract: bool=False, mro: list[TypeInfo] | None=None, bases: list[Instance] | None=None, typevars: list[str] | None=None, typevar_tuple_index: int | None=None, variances: list[int] | None=None) -> TypeInfo:
        if False:
            for i in range(10):
                print('nop')
        'Make a TypeInfo suitable for use in unit tests.'
        class_def = ClassDef(name, Block([]), None, [])
        class_def.fullname = name
        if module_name is None:
            if '.' in name:
                module_name = name.rsplit('.', 1)[0]
            else:
                module_name = '__main__'
        if typevars:
            v: list[TypeVarLikeType] = []
            for (id, n) in enumerate(typevars, 1):
                if typevar_tuple_index is not None and id - 1 == typevar_tuple_index:
                    v.append(TypeVarTupleType(n, n, id, self.std_tuple.copy_modified(args=[self.o]), self.std_tuple.copy_modified(args=[self.o]), AnyType(TypeOfAny.from_omitted_generics)))
                else:
                    if variances:
                        variance = variances[id - 1]
                    else:
                        variance = COVARIANT
                    v.append(TypeVarType(n, n, id, [], self.o, AnyType(TypeOfAny.from_omitted_generics), variance=variance))
            class_def.type_vars = v
        info = TypeInfo(SymbolTable(), class_def, module_name)
        if mro is None:
            mro = []
            if name != 'builtins.object':
                mro.append(self.oi)
        info.mro = [info] + mro
        if bases is None:
            if mro:
                bases = [Instance(mro[0], [])]
            else:
                bases = []
        info.bases = bases
        return info

    def def_alias_1(self, base: Instance) -> tuple[TypeAliasType, Type]:
        if False:
            print('Hello World!')
        A = TypeAliasType(None, [])
        target = Instance(self.std_tuplei, [UnionType([base, A])])
        AN = TypeAlias(target, '__main__.A', -1, -1)
        A.alias = AN
        return (A, target)

    def def_alias_2(self, base: Instance) -> tuple[TypeAliasType, Type]:
        if False:
            while True:
                i = 10
        A = TypeAliasType(None, [])
        target = UnionType([base, Instance(self.std_tuplei, [A])])
        AN = TypeAlias(target, '__main__.A', -1, -1)
        A.alias = AN
        return (A, target)

    def non_rec_alias(self, target: Type, alias_tvars: list[TypeVarLikeType] | None=None, args: list[Type] | None=None) -> TypeAliasType:
        if False:
            i = 10
            return i + 15
        AN = TypeAlias(target, '__main__.A', -1, -1, alias_tvars=alias_tvars)
        if args is None:
            args = []
        return TypeAliasType(AN, args)

class InterfaceTypeFixture(TypeFixture):
    """Extension of TypeFixture that contains additional generic
    interface types."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.gfi = self.make_type_info('GF', typevars=['T'], is_abstract=True)
        self.m1i = self.make_type_info('M1', is_abstract=True, mro=[self.gfi, self.oi], bases=[Instance(self.gfi, [self.a])])
        self.gfa = Instance(self.gfi, [self.a])
        self.gfb = Instance(self.gfi, [self.b])
        self.m1 = Instance(self.m1i, [])