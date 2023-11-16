from __future__ import annotations
from mypy.nodes import ParamSpecExpr, SymbolTableNode, TypeVarExpr, TypeVarLikeExpr, TypeVarTupleExpr
from mypy.types import ParamSpecFlavor, ParamSpecType, TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType

class TypeVarLikeScope:
    """Scope that holds bindings for type variables and parameter specifications.

    Node fullname -> TypeVarLikeType.
    """

    def __init__(self, parent: TypeVarLikeScope | None=None, is_class_scope: bool=False, prohibited: TypeVarLikeScope | None=None, namespace: str='') -> None:
        if False:
            while True:
                i = 10
        "Initializer for TypeVarLikeScope\n\n        Parameters:\n          parent: the outer scope for this scope\n          is_class_scope: True if this represents a generic class\n          prohibited: Type variables that aren't strictly in scope exactly,\n                      but can't be bound because they're part of an outer class's scope.\n        "
        self.scope: dict[str, TypeVarLikeType] = {}
        self.parent = parent
        self.func_id = 0
        self.class_id = 0
        self.is_class_scope = is_class_scope
        self.prohibited = prohibited
        self.namespace = namespace
        if parent is not None:
            self.func_id = parent.func_id
            self.class_id = parent.class_id

    def get_function_scope(self) -> TypeVarLikeScope | None:
        if False:
            i = 10
            return i + 15
        "Get the nearest parent that's a function scope, not a class scope"
        it: TypeVarLikeScope | None = self
        while it is not None and it.is_class_scope:
            it = it.parent
        return it

    def allow_binding(self, fullname: str) -> bool:
        if False:
            print('Hello World!')
        if fullname in self.scope:
            return False
        elif self.parent and (not self.parent.allow_binding(fullname)):
            return False
        elif self.prohibited and (not self.prohibited.allow_binding(fullname)):
            return False
        return True

    def method_frame(self) -> TypeVarLikeScope:
        if False:
            i = 10
            return i + 15
        'A new scope frame for binding a method'
        return TypeVarLikeScope(self, False, None)

    def class_frame(self, namespace: str) -> TypeVarLikeScope:
        if False:
            return 10
        "A new scope frame for binding a class. Prohibits *this* class's tvars"
        return TypeVarLikeScope(self.get_function_scope(), True, self, namespace=namespace)

    def new_unique_func_id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Used by plugin-like code that needs to make synthetic generic functions.'
        self.func_id -= 1
        return self.func_id

    def bind_new(self, name: str, tvar_expr: TypeVarLikeExpr) -> TypeVarLikeType:
        if False:
            return 10
        if self.is_class_scope:
            self.class_id += 1
            i = self.class_id
            namespace = self.namespace
        else:
            self.func_id -= 1
            i = self.func_id
            namespace = ''
        if isinstance(tvar_expr, TypeVarExpr):
            tvar_def: TypeVarLikeType = TypeVarType(name=name, fullname=tvar_expr.fullname, id=TypeVarId(i, namespace=namespace), values=tvar_expr.values, upper_bound=tvar_expr.upper_bound, default=tvar_expr.default, variance=tvar_expr.variance, line=tvar_expr.line, column=tvar_expr.column)
        elif isinstance(tvar_expr, ParamSpecExpr):
            tvar_def = ParamSpecType(name, tvar_expr.fullname, i, flavor=ParamSpecFlavor.BARE, upper_bound=tvar_expr.upper_bound, default=tvar_expr.default, line=tvar_expr.line, column=tvar_expr.column)
        elif isinstance(tvar_expr, TypeVarTupleExpr):
            tvar_def = TypeVarTupleType(name, tvar_expr.fullname, i, upper_bound=tvar_expr.upper_bound, tuple_fallback=tvar_expr.tuple_fallback, default=tvar_expr.default, line=tvar_expr.line, column=tvar_expr.column)
        else:
            assert False
        self.scope[tvar_expr.fullname] = tvar_def
        return tvar_def

    def bind_existing(self, tvar_def: TypeVarLikeType) -> None:
        if False:
            while True:
                i = 10
        self.scope[tvar_def.fullname] = tvar_def

    def get_binding(self, item: str | SymbolTableNode) -> TypeVarLikeType | None:
        if False:
            for i in range(10):
                print('nop')
        fullname = item.fullname if isinstance(item, SymbolTableNode) else item
        assert fullname
        if fullname in self.scope:
            return self.scope[fullname]
        elif self.parent is not None:
            return self.parent.get_binding(fullname)
        else:
            return None

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        me = ', '.join((f'{k}: {v.name}`{v.id}' for (k, v) in self.scope.items()))
        if self.parent is None:
            return me
        return f'{self.parent} <- {me}'