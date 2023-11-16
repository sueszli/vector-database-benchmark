"""Calculate some properties of classes.

These happen after semantic analysis and before type checking.
"""
from __future__ import annotations
from typing import Final
from mypy.errors import Errors
from mypy.nodes import IMPLICITLY_ABSTRACT, IS_ABSTRACT, CallExpr, Decorator, FuncDef, Node, OverloadedFuncDef, PromoteExpr, SymbolTable, TypeInfo, Var
from mypy.options import Options
from mypy.types import MYPYC_NATIVE_INT_NAMES, Instance, ProperType
TYPE_PROMOTIONS: Final = {'builtins.int': 'float', 'builtins.float': 'complex', 'builtins.bytearray': 'bytes', 'builtins.memoryview': 'bytes'}

def calculate_class_abstract_status(typ: TypeInfo, is_stub_file: bool, errors: Errors) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Calculate abstract status of a class.\n\n    Set is_abstract of the type to True if the type has an unimplemented\n    abstract attribute.  Also compute a list of abstract attributes.\n    Report error is required ABCMeta metaclass is missing.\n    '
    typ.is_abstract = False
    typ.abstract_attributes = []
    if typ.typeddict_type:
        return
    concrete: set[str] = set()
    abstract: list[tuple[str, int]] = []
    abstract_in_this_class: list[str] = []
    if typ.is_newtype:
        return
    for base in typ.mro:
        for (name, symnode) in base.names.items():
            node = symnode.node
            if isinstance(node, OverloadedFuncDef):
                if node.items:
                    func: Node | None = node.items[0]
                else:
                    func = None
            else:
                func = node
            if isinstance(func, Decorator):
                func = func.func
            if isinstance(func, FuncDef):
                if func.abstract_status in (IS_ABSTRACT, IMPLICITLY_ABSTRACT) and name not in concrete:
                    typ.is_abstract = True
                    abstract.append((name, func.abstract_status))
                    if base is typ:
                        abstract_in_this_class.append(name)
            elif isinstance(node, Var):
                if node.is_abstract_var and name not in concrete:
                    typ.is_abstract = True
                    abstract.append((name, IS_ABSTRACT))
                    if base is typ:
                        abstract_in_this_class.append(name)
            concrete.add(name)
    typ.abstract_attributes = sorted(abstract)
    if is_stub_file:
        if typ.declared_metaclass and typ.declared_metaclass.type.has_base('abc.ABCMeta'):
            return
        if typ.is_protocol:
            return
        if abstract and (not abstract_in_this_class):

            def report(message: str, severity: str) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                errors.report(typ.line, typ.column, message, severity=severity)
            attrs = ', '.join((f'"{attr}"' for (attr, _) in sorted(abstract)))
            report(f'Class {typ.fullname} has abstract attributes {attrs}', 'error')
            report("If it is meant to be abstract, add 'abc.ABCMeta' as an explicit metaclass", 'note')
    if typ.is_final and abstract:
        attrs = ', '.join((f'"{attr}"' for (attr, _) in sorted(abstract)))
        errors.report(typ.line, typ.column, f'Final class {typ.fullname} has abstract attributes {attrs}')

def check_protocol_status(info: TypeInfo, errors: Errors) -> None:
    if False:
        while True:
            i = 10
    'Check that all classes in MRO of a protocol are protocols'
    if info.is_protocol:
        for type in info.bases:
            if not type.type.is_protocol and type.type.fullname != 'builtins.object':

                def report(message: str, severity: str) -> None:
                    if False:
                        print('Hello World!')
                    errors.report(info.line, info.column, message, severity=severity)
                report('All bases of a protocol must be protocols', 'error')

def calculate_class_vars(info: TypeInfo) -> None:
    if False:
        i = 10
        return i + 15
    'Try to infer additional class variables.\n\n    Subclass attribute assignments with no type annotation are assumed\n    to be classvar if overriding a declared classvar from the base\n    class.\n\n    This must happen after the main semantic analysis pass, since\n    this depends on base class bodies having been fully analyzed.\n    '
    for (name, sym) in info.names.items():
        node = sym.node
        if isinstance(node, Var) and node.info and node.is_inferred and (not node.is_classvar):
            for base in info.mro[1:]:
                member = base.names.get(name)
                if member is not None and isinstance(member.node, Var) and member.node.is_classvar:
                    node.is_classvar = True

def add_type_promotion(info: TypeInfo, module_names: SymbolTable, options: Options, builtin_names: SymbolTable) -> None:
    if False:
        while True:
            i = 10
    "Setup extra, ad-hoc subtyping relationships between classes (promotion).\n\n    This includes things like 'int' being compatible with 'float'.\n    "
    defn = info.defn
    promote_targets: list[ProperType] = []
    for decorator in defn.decorators:
        if isinstance(decorator, CallExpr):
            analyzed = decorator.analyzed
            if isinstance(analyzed, PromoteExpr):
                promote_targets.append(analyzed.type)
    if not promote_targets:
        if defn.fullname in TYPE_PROMOTIONS:
            target_sym = module_names.get(TYPE_PROMOTIONS[defn.fullname])
            if defn.fullname == 'builtins.bytearray' and options.disable_bytearray_promotion:
                target_sym = None
            elif defn.fullname == 'builtins.memoryview' and options.disable_memoryview_promotion:
                target_sym = None
            if target_sym:
                target_info = target_sym.node
                assert isinstance(target_info, TypeInfo)
                promote_targets.append(Instance(target_info, []))
    if defn.fullname in MYPYC_NATIVE_INT_NAMES:
        int_sym = builtin_names['int']
        assert isinstance(int_sym.node, TypeInfo)
        int_sym.node._promote.append(Instance(defn.info, []))
        defn.info.alt_promote = Instance(int_sym.node, [])
    if promote_targets:
        defn.info._promote.extend(promote_targets)