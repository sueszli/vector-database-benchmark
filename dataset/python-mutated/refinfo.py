"""Find line-level reference information from a mypy AST (undocumented feature)"""
from __future__ import annotations
from mypy.nodes import LDEF, Expression, FuncDef, MemberExpr, MypyFile, NameExpr, RefExpr, SymbolNode, TypeInfo
from mypy.traverser import TraverserVisitor
from mypy.typeops import tuple_fallback
from mypy.types import FunctionLike, Instance, TupleType, Type, TypeType, TypeVarLikeType, get_proper_type

class RefInfoVisitor(TraverserVisitor):

    def __init__(self, type_map: dict[Expression, Type]) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.type_map = type_map
        self.data: list[dict[str, object]] = []

    def visit_name_expr(self, expr: NameExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().visit_name_expr(expr)
        self.record_ref_expr(expr)

    def visit_member_expr(self, expr: MemberExpr) -> None:
        if False:
            i = 10
            return i + 15
        super().visit_member_expr(expr)
        self.record_ref_expr(expr)

    def visit_func_def(self, func: FuncDef) -> None:
        if False:
            i = 10
            return i + 15
        if func.expanded:
            for item in func.expanded:
                if isinstance(item, FuncDef):
                    super().visit_func_def(item)
        else:
            super().visit_func_def(func)

    def record_ref_expr(self, expr: RefExpr) -> None:
        if False:
            for i in range(10):
                print('nop')
        fullname = None
        if expr.kind != LDEF and '.' in expr.fullname:
            fullname = expr.fullname
        elif isinstance(expr, MemberExpr):
            typ = self.type_map.get(expr.expr)
            sym = None
            if isinstance(expr.expr, RefExpr):
                sym = expr.expr.node
            if typ:
                tfn = type_fullname(typ, sym)
                if tfn:
                    fullname = f'{tfn}.{expr.name}'
            if not fullname:
                fullname = f'*.{expr.name}'
        if fullname is not None:
            self.data.append({'line': expr.line, 'column': expr.column, 'target': fullname})

def type_fullname(typ: Type, node: SymbolNode | None=None) -> str | None:
    if False:
        while True:
            i = 10
    typ = get_proper_type(typ)
    if isinstance(typ, Instance):
        return typ.type.fullname
    elif isinstance(typ, TypeType):
        return type_fullname(typ.item)
    elif isinstance(typ, FunctionLike) and typ.is_type_obj():
        if isinstance(node, TypeInfo):
            return node.fullname
        return type_fullname(typ.fallback)
    elif isinstance(typ, TupleType):
        return type_fullname(tuple_fallback(typ))
    elif isinstance(typ, TypeVarLikeType):
        return type_fullname(typ.upper_bound)
    return None

def get_undocumented_ref_info_json(tree: MypyFile, type_map: dict[Expression, Type]) -> list[dict[str, object]]:
    if False:
        for i in range(10):
            print('nop')
    visitor = RefInfoVisitor(type_map)
    tree.accept(visitor)
    return visitor.data