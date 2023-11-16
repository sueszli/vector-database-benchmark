from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import Decorator
from mypy.nodes import LambdaExpr
from mypy.nodes import ListExpr
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import PlaceholderNode
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolNode
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import AnyType
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneType
from mypy.types import ProperType
from mypy.types import Type
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType
from . import apply
from . import infer
from . import names
from . import util

def scan_declarative_assignments_and_apply_types(cls: ClassDef, api: SemanticAnalyzerPluginInterface, is_mixin_scan: bool=False) -> Optional[List[util.SQLAlchemyAttribute]]:
    if False:
        for i in range(10):
            print('nop')
    info = util.info_for_cls(cls, api)
    if info is None:
        return None
    elif cls.fullname.startswith('builtins'):
        return None
    mapped_attributes: Optional[List[util.SQLAlchemyAttribute]] = util.get_mapped_attributes(info, api)
    util.establish_as_sqlalchemy(info)
    if mapped_attributes is not None:
        if not is_mixin_scan:
            apply.re_apply_declarative_assignments(cls, api, mapped_attributes)
        return mapped_attributes
    mapped_attributes = []
    if not cls.defs.body:
        for (sym_name, sym) in info.names.items():
            _scan_symbol_table_entry(cls, api, sym_name, sym, mapped_attributes)
    else:
        for stmt in util.flatten_typechecking(cls.defs.body):
            if isinstance(stmt, AssignmentStmt):
                _scan_declarative_assignment_stmt(cls, api, stmt, mapped_attributes)
            elif isinstance(stmt, Decorator):
                _scan_declarative_decorator_stmt(cls, api, stmt, mapped_attributes)
    _scan_for_mapped_bases(cls, api)
    if not is_mixin_scan:
        apply.add_additional_orm_attributes(cls, api, mapped_attributes)
    util.set_mapped_attributes(info, mapped_attributes)
    return mapped_attributes

def _scan_symbol_table_entry(cls: ClassDef, api: SemanticAnalyzerPluginInterface, name: str, value: SymbolTableNode, attributes: List[util.SQLAlchemyAttribute]) -> None:
    if False:
        print('Hello World!')
    "Extract mapping information from a SymbolTableNode that's in the\n    type.names dictionary.\n\n    "
    value_type = get_proper_type(value.type)
    if not isinstance(value_type, Instance):
        return
    left_hand_explicit_type = None
    type_id = names.type_id_for_named_node(value_type.type)
    err = False
    if type_id in {names.MAPPED, names.RELATIONSHIP, names.COMPOSITE_PROPERTY, names.MAPPER_PROPERTY, names.SYNONYM_PROPERTY, names.COLUMN_PROPERTY}:
        if value_type.args:
            left_hand_explicit_type = get_proper_type(value_type.args[0])
        else:
            err = True
    elif type_id is names.COLUMN:
        if not value_type.args:
            err = True
        else:
            typeengine_arg: Union[ProperType, TypeInfo] = get_proper_type(value_type.args[0])
            if isinstance(typeengine_arg, Instance):
                typeengine_arg = typeengine_arg.type
            if isinstance(typeengine_arg, (UnboundType, TypeInfo)):
                sym = api.lookup_qualified(typeengine_arg.name, typeengine_arg)
                if sym is not None and isinstance(sym.node, TypeInfo):
                    if names.has_base_type_id(sym.node, names.TYPEENGINE):
                        left_hand_explicit_type = UnionType([infer.extract_python_type_from_typeengine(api, sym.node, []), NoneType()])
                    else:
                        util.fail(api, "Column type should be a TypeEngine subclass not '{}'".format(sym.node.fullname), value_type)
    if err:
        msg = "Can't infer type from attribute {} on class {}. please specify a return type from this function that is one of: Mapped[<python type>], relationship[<target class>], Column[<TypeEngine>], MapperProperty[<python type>]"
        util.fail(api, msg.format(name, cls.name), cls)
        left_hand_explicit_type = AnyType(TypeOfAny.special_form)
    if left_hand_explicit_type is not None:
        assert value.node is not None
        attributes.append(util.SQLAlchemyAttribute(name=name, line=value.node.line, column=value.node.column, typ=left_hand_explicit_type, info=cls.info))

def _scan_declarative_decorator_stmt(cls: ClassDef, api: SemanticAnalyzerPluginInterface, stmt: Decorator, attributes: List[util.SQLAlchemyAttribute]) -> None:
    if False:
        i = 10
        return i + 15
    'Extract mapping information from a @declared_attr in a declarative\n    class.\n\n    E.g.::\n\n        @reg.mapped\n        class MyClass:\n            # ...\n\n            @declared_attr\n            def updated_at(cls) -> Column[DateTime]:\n                return Column(DateTime)\n\n    Will resolve in mypy as::\n\n        @reg.mapped\n        class MyClass:\n            # ...\n\n            updated_at: Mapped[Optional[datetime.datetime]]\n\n    '
    for dec in stmt.decorators:
        if isinstance(dec, (NameExpr, MemberExpr, SymbolNode)) and names.type_id_for_named_node(dec) is names.DECLARED_ATTR:
            break
    else:
        return
    dec_index = cls.defs.body.index(stmt)
    left_hand_explicit_type: Optional[ProperType] = None
    if util.name_is_dunder(stmt.name):
        any_ = AnyType(TypeOfAny.special_form)
        left_node = NameExpr(stmt.var.name)
        left_node.node = stmt.var
        new_stmt = AssignmentStmt([left_node], TempNode(any_))
        new_stmt.type = left_node.node.type
        cls.defs.body[dec_index] = new_stmt
        return
    elif isinstance(stmt.func.type, CallableType):
        func_type = stmt.func.type.ret_type
        if isinstance(func_type, UnboundType):
            type_id = names.type_id_for_unbound_type(func_type, cls, api)
        else:
            return
        if type_id in {names.MAPPED, names.RELATIONSHIP, names.COMPOSITE_PROPERTY, names.MAPPER_PROPERTY, names.SYNONYM_PROPERTY, names.COLUMN_PROPERTY} and func_type.args:
            left_hand_explicit_type = get_proper_type(func_type.args[0])
        elif type_id is names.COLUMN and func_type.args:
            typeengine_arg = func_type.args[0]
            if isinstance(typeengine_arg, UnboundType):
                sym = api.lookup_qualified(typeengine_arg.name, typeengine_arg)
                if sym is not None and isinstance(sym.node, TypeInfo):
                    if names.has_base_type_id(sym.node, names.TYPEENGINE):
                        left_hand_explicit_type = UnionType([infer.extract_python_type_from_typeengine(api, sym.node, []), NoneType()])
                    else:
                        util.fail(api, "Column type should be a TypeEngine subclass not '{}'".format(sym.node.fullname), func_type)
    if left_hand_explicit_type is None:
        msg = "Can't infer type from @declared_attr on function '{}';  please specify a return type from this function that is one of: Mapped[<python type>], relationship[<target class>], Column[<TypeEngine>], MapperProperty[<python type>]"
        util.fail(api, msg.format(stmt.var.name), stmt)
        left_hand_explicit_type = AnyType(TypeOfAny.special_form)
    left_node = NameExpr(stmt.var.name)
    left_node.node = stmt.var
    if isinstance(left_hand_explicit_type, UnboundType):
        left_hand_explicit_type = get_proper_type(util.unbound_to_instance(api, left_hand_explicit_type))
    left_node.node.type = api.named_type(names.NAMED_TYPE_SQLA_MAPPED, [left_hand_explicit_type])
    rvalue = names.expr_to_mapped_constructor(LambdaExpr(stmt.func.arguments, stmt.func.body))
    new_stmt = AssignmentStmt([left_node], rvalue)
    new_stmt.type = left_node.node.type
    attributes.append(util.SQLAlchemyAttribute(name=left_node.name, line=stmt.line, column=stmt.column, typ=left_hand_explicit_type, info=cls.info))
    cls.defs.body[dec_index] = new_stmt

def _scan_declarative_assignment_stmt(cls: ClassDef, api: SemanticAnalyzerPluginInterface, stmt: AssignmentStmt, attributes: List[util.SQLAlchemyAttribute]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Extract mapping information from an assignment statement in a\n    declarative class.\n\n    '
    lvalue = stmt.lvalues[0]
    if not isinstance(lvalue, NameExpr):
        return
    sym = cls.info.names.get(lvalue.name)
    assert sym is not None
    node = sym.node
    if isinstance(node, PlaceholderNode):
        return
    assert node is lvalue.node
    assert isinstance(node, Var)
    if node.name == '__abstract__':
        if api.parse_bool(stmt.rvalue) is True:
            util.set_is_base(cls.info)
        return
    elif node.name == '__tablename__':
        util.set_has_table(cls.info)
    elif node.name.startswith('__'):
        return
    elif node.name == '_mypy_mapped_attrs':
        if not isinstance(stmt.rvalue, ListExpr):
            util.fail(api, '_mypy_mapped_attrs is expected to be a list', stmt)
        else:
            for item in stmt.rvalue.items:
                if isinstance(item, (NameExpr, StrExpr)):
                    apply.apply_mypy_mapped_attr(cls, api, item, attributes)
    left_hand_mapped_type: Optional[Type] = None
    left_hand_explicit_type: Optional[ProperType] = None
    if node.is_inferred or node.type is None:
        if isinstance(stmt.type, UnboundType):
            left_hand_explicit_type = stmt.type
            if stmt.type.name == 'Mapped':
                mapped_sym = api.lookup_qualified('Mapped', cls)
                if mapped_sym is not None and mapped_sym.node is not None and (names.type_id_for_named_node(mapped_sym.node) is names.MAPPED):
                    left_hand_explicit_type = get_proper_type(stmt.type.args[0])
                    left_hand_mapped_type = stmt.type
    else:
        node_type = get_proper_type(node.type)
        if isinstance(node_type, Instance) and names.type_id_for_named_node(node_type.type) is names.MAPPED:
            left_hand_explicit_type = get_proper_type(node_type.args[0])
            left_hand_mapped_type = node_type
        else:
            left_hand_explicit_type = node_type
            left_hand_mapped_type = None
    if isinstance(stmt.rvalue, TempNode) and left_hand_mapped_type is not None:
        python_type_for_type = left_hand_explicit_type
    elif isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue.callee, RefExpr):
        python_type_for_type = infer.infer_type_from_right_hand_nameexpr(api, stmt, node, left_hand_explicit_type, stmt.rvalue.callee)
        if python_type_for_type is None:
            return
    else:
        return
    assert python_type_for_type is not None
    attributes.append(util.SQLAlchemyAttribute(name=node.name, line=stmt.line, column=stmt.column, typ=python_type_for_type, info=cls.info))
    apply.apply_type_to_mapped_statement(api, stmt, lvalue, left_hand_explicit_type, python_type_for_type)

def _scan_for_mapped_bases(cls: ClassDef, api: SemanticAnalyzerPluginInterface) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Given a class, iterate through its superclass hierarchy to find\n    all other classes that are considered as ORM-significant.\n\n    Locates non-mapped mixins and scans them for mapped attributes to be\n    applied to subclasses.\n\n    '
    info = util.info_for_cls(cls, api)
    if info is None:
        return
    for base_info in info.mro[1:-1]:
        if base_info.fullname.startswith('builtins'):
            continue
        scan_declarative_assignments_and_apply_types(base_info.defn, api, is_mixin_scan=True)