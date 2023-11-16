from __future__ import annotations
import ast
from ast import AST, Attribute, BinOp, Call, ClassDef, Constant, Expression, Name, Subscript
from contextlib import nullcontext
from enum import Enum
from typing import cast, ContextManager, Dict, List, Optional, overload, Set, Tuple, Type, TYPE_CHECKING, Union
from ..errors import TypedSyntaxError
from ..symbols import ModuleScope, Scope
from .types import Callable, Class, ClassVar, CType, DataclassDecorator, DynamicClass, ExactClass, FinalClass, Function, FunctionGroup, InitVar, KnownBoolean, MethodType, NativeDecorator, TType, TypeDescr, UnionType, UnknownDecoratedMethod, Value
from .visitor import GenericVisitor
if TYPE_CHECKING:
    from .compiler import Compiler

class ModuleFlag(Enum):
    CHECKED_DICTS = 1
    CHECKED_LISTS = 3

class ModuleTableException(Exception):
    pass

class ReferenceVisitor(GenericVisitor[Optional[Value]]):

    def __init__(self, module: ModuleTable) -> None:
        if False:
            return 10
        super().__init__(module)
        self.types: Dict[AST, Value] = {}
        self.subscr_nesting = 0
        self.local_names: Dict[str, Value] = {}

    def visitName(self, node: Name) -> Optional[Value]:
        if False:
            for i in range(10):
                print('nop')
        if node.id in self.local_names:
            return self.local_names[node.id]
        return self.module.get_child(node.id) or self.module.compiler.builtins.get_child(node.id)

    def visitAttribute(self, node: Attribute) -> Optional[Value]:
        if False:
            print('Hello World!')
        val = self.visit(node.value)
        if val is not None:
            return val.resolve_attr(node, self)

    def add_local_name(self, name: str, value: Value) -> None:
        if False:
            return 10
        self.local_names[name] = value

    def clear_local_names(self) -> None:
        if False:
            while True:
                i = 10
        self.local_names = {}

class AnnotationVisitor(ReferenceVisitor):

    def resolve_annotation(self, node: ast.AST, *, is_declaration: bool=False) -> Optional[Class]:
        if False:
            while True:
                i = 10
        with self.error_context(node):
            klass = self.visit(node)
            if not isinstance(klass, Class):
                return None
            if self.subscr_nesting or not is_declaration:
                if isinstance(klass, FinalClass):
                    raise TypedSyntaxError('Final annotation is only valid in initial declaration of attribute or module-level constant')
                if isinstance(klass, ClassVar):
                    raise TypedSyntaxError('ClassVar is allowed only in class attribute annotations. Class Finals are inferred ClassVar; do not nest with Final.')
                if isinstance(klass, InitVar):
                    raise TypedSyntaxError('InitVar is allowed only in class attribute annotations.')
            if isinstance(klass, ExactClass):
                klass = klass.unwrap().exact_type()
            elif isinstance(klass, FinalClass):
                pass
            else:
                klass = klass.inexact_type()
            if klass.unwrap() is self.type_env.float:
                klass = self.compiler.type_env.get_union((self.type_env.float, self.type_env.int))
            if isinstance(klass, UnionType) and klass is not self.type_env.union and (klass is not self.type_env.optional) and (klass.opt_type is None):
                return None
            return klass

    def visitSubscript(self, node: Subscript) -> Optional[Value]:
        if False:
            print('Hello World!')
        target = self.resolve_annotation(node.value, is_declaration=True)
        if target is None:
            return None
        self.subscr_nesting += 1
        slice = self.visit(node.slice) or self.type_env.DYNAMIC
        self.subscr_nesting -= 1
        return target.resolve_subscr(node, slice, self) or target

    def visitBinOp(self, node: BinOp) -> Optional[Value]:
        if False:
            i = 10
            return i + 15
        if isinstance(node.op, ast.BitOr):
            ltype = self.resolve_annotation(node.left)
            rtype = self.resolve_annotation(node.right)
            if ltype is None or rtype is None:
                return None
            return self.module.compiler.type_env.get_union((ltype, rtype))

    def visitConstant(self, node: Constant) -> Optional[Value]:
        if False:
            print('Hello World!')
        sval = node.value
        if sval is None:
            return self.type_env.none
        elif isinstance(sval, str):
            n = cast(Expression, ast.parse(node.value, '', 'eval')).body
            return self.visit(n)

class DeferredValue:

    def __init__(self, mod_name: str, name: str, compiler: Compiler) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.mod_name = mod_name
        self.name = name
        self.compiler = compiler

    def resolve(self) -> Value:
        if False:
            print('Hello World!')
        mod = self.compiler.modules.get(self.mod_name)
        if mod is not None:
            val = mod.get_child(self.name)
            if val is not None:
                return val
        return self.compiler.type_env.DYNAMIC

class ModuleTable:

    def __init__(self, name: str, filename: str, compiler: Compiler, members: Optional[Dict[str, Value]]=None) -> None:
        if False:
            while True:
                i = 10
        self.name = name
        self.filename = filename
        self._children: Dict[str, Value | DeferredValue] = {}
        if members is not None:
            self._children.update(members)
        self.compiler = compiler
        self.types: Dict[AST, Value] = {}
        self.node_data: Dict[Tuple[AST, object], object] = {}
        self.flags: Set[ModuleFlag] = set()
        self.decls: List[Tuple[AST, Optional[str], Optional[Value]]] = []
        self.implicit_decl_names: Set[str] = set()
        self.compile_non_static: Set[AST] = set()
        self.imported_from: Dict[str, Tuple[str, str]] = {}
        self.named_finals: Dict[str, ast.Constant] = {}
        self.first_pass_done = False
        self.ann_visitor = AnnotationVisitor(self)
        self.ref_visitor = ReferenceVisitor(self)

    @overload
    def get_child(self, name: str, default: Value=...) -> Value:
        if False:
            i = 10
            return i + 15
        ...

    def get_child(self, name: str, default: Optional[Value]=None) -> Optional[Value]:
        if False:
            while True:
                i = 10
        res = self._children.get(name, default)
        if isinstance(res, DeferredValue):
            self._children[name] = res = res.resolve()
        return res

    def syntax_error(self, msg: str, node: AST) -> None:
        if False:
            i = 10
            return i + 15
        return self.compiler.error_sink.syntax_error(msg, self.filename, node)

    def error_context(self, node: Optional[AST]) -> ContextManager[None]:
        if False:
            print('Hello World!')
        if node is None:
            return nullcontext()
        return self.compiler.error_sink.error_context(self.filename, node)

    def declare_class(self, node: ClassDef, klass: Class) -> None:
        if False:
            print('Hello World!')
        if self.first_pass_done:
            raise ModuleTableException('Attempted to declare a class after the declaration visit')
        self.decls.append((node, node.name, klass))
        self._children[node.name] = klass

    def declare_function(self, func: Function) -> None:
        if False:
            print('Hello World!')
        if self.first_pass_done:
            raise ModuleTableException('Attempted to declare a function after the declaration visit')
        existing = self._children.get(func.func_name)
        new_member = func
        if existing is not None:
            if isinstance(existing, Function):
                new_member = FunctionGroup([existing, new_member], func.klass.type_env)
            elif isinstance(existing, FunctionGroup):
                existing.functions.append(new_member)
                new_member = existing
            else:
                raise TypedSyntaxError(f'function conflicts with other member {func.func_name} in {self.name}')
        self.decls.append((func.node, func.func_name, new_member))
        self._children[func.func_name] = new_member

    def _get_inferred_type(self, value: ast.expr) -> Optional[Value]:
        if False:
            i = 10
            return i + 15
        if not isinstance(value, ast.Name):
            return None
        return self.get_child(value.id)

    def finish_bind(self) -> None:
        if False:
            while True:
                i = 10
        self.first_pass_done = True
        for (node, name, value) in self.decls:
            with self.error_context(node):
                if value is not None:
                    assert name is not None
                    new_value = value.finish_bind(self, None)
                    if new_value is None:
                        if isinstance(self.types[node], UnknownDecoratedMethod):
                            self._children[name] = self.compiler.type_env.DYNAMIC
                        else:
                            del self._children[name]
                    elif new_value is not value:
                        self._children[name] = new_value
                if isinstance(node, ast.AnnAssign):
                    typ = self.resolve_annotation(node.annotation, is_declaration=True)
                    is_final_dynamic = False
                    if typ is not None:
                        target = node.target
                        instance = typ.instance
                        value = node.value
                        is_final_dynamic = False
                        if value is not None and isinstance(typ, FinalClass) and isinstance(typ.unwrap(), DynamicClass):
                            is_final_dynamic = True
                            instance = self._get_inferred_type(value) or typ.unwrap().instance
                        if isinstance(target, ast.Name):
                            if not is_final_dynamic and isinstance(typ, FinalClass):
                                instance = typ.unwrap().instance
                            self._children[target.id] = instance
                    if isinstance(typ, FinalClass):
                        target = node.target
                        value = node.value
                        if not value:
                            raise TypedSyntaxError('Must assign a value when declaring a Final')
                        elif not isinstance(typ, CType) and isinstance(target, ast.Name) and isinstance(value, ast.Constant):
                            self.named_finals[target.id] = value
        for name in self.implicit_decl_names:
            if name not in self._children:
                self._children[name] = self.compiler.type_env.DYNAMIC
        self.decls.clear()
        self.implicit_decl_names.clear()

    def resolve_type(self, node: ast.AST) -> Optional[Class]:
        if False:
            print('Hello World!')
        typ = self.ann_visitor.visit(node)
        if isinstance(typ, Class):
            return typ

    def resolve_decorator(self, node: ast.AST) -> Optional[Value]:
        if False:
            print('Hello World!')
        if isinstance(node, Call):
            func = self.ref_visitor.visit(node.func)
            if isinstance(func, Class):
                return func.instance
            elif isinstance(func, DataclassDecorator):
                return func
            elif isinstance(func, NativeDecorator):
                return func
            elif isinstance(func, Callable):
                return func.return_type.resolved().instance
            elif isinstance(func, MethodType):
                return func.function.return_type.resolved().instance
        return self.ref_visitor.visit(node)

    def resolve_annotation(self, node: ast.AST, *, is_declaration: bool=False) -> Optional[Class]:
        if False:
            print('Hello World!')
        assert self.first_pass_done, 'Type annotations cannot be resolved until after initial pass, so that all imports and types are available.'
        return self.ann_visitor.resolve_annotation(node, is_declaration=is_declaration)

    def resolve_name_with_descr(self, name: str) -> Tuple[Optional[Value], Optional[TypeDescr]]:
        if False:
            return 10
        if (val := self.get_child(name)):
            return (val, (self.name, name))
        elif (val := self.compiler.builtins.get_child(name)):
            return (val, None)
        return (None, None)

    def resolve_name(self, name: str) -> Optional[Value]:
        if False:
            for i in range(10):
                print('nop')
        return self.resolve_name_with_descr(name)[0]

    def get_final_literal(self, node: AST, scope: Scope) -> Optional[ast.Constant]:
        if False:
            return 10
        if not isinstance(node, Name):
            return None
        final_val = self.named_finals.get(node.id, None)
        if final_val is not None and isinstance(node.ctx, ast.Load) and (isinstance(scope, ModuleScope) or node.id not in scope.defs):
            return final_val

    def declare_import(self, name: str, source: Tuple[str, str] | None, val: Value | DeferredValue) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Declare a name imported into this module.\n\n        `name` is the name in this module's namespace. `source` is a (str, str)\n        tuple of (source_module, source_name) for an `import from`. For a\n        top-level module import, `source` should be `None`.\n\n        "
        if self.first_pass_done:
            raise ModuleTableException('Attempted to declare an import after the declaration visit')
        self._children[name] = val
        if source is not None:
            self.imported_from[name] = source

    def declare_variable(self, node: ast.AnnAssign, module: ModuleTable) -> None:
        if False:
            return 10
        self.decls.append((node, None, None))

    def declare_variables(self, node: ast.Assign, module: ModuleTable) -> None:
        if False:
            while True:
                i = 10
        targets = node.targets
        for target in targets:
            if isinstance(target, ast.Name):
                self.implicit_decl_names.add(target.id)

    def get_node_data(self, key: AST, data_type: Type[TType]) -> TType:
        if False:
            while True:
                i = 10
        return cast(TType, self.node_data[key, data_type])

    def get_opt_node_data(self, key: AST, data_type: Type[TType]) -> TType | None:
        if False:
            for i in range(10):
                print('nop')
        return cast(Optional[TType], self.node_data.get((key, data_type)))

    def set_node_data(self, key: AST, data_type: Type[TType], value: TType) -> None:
        if False:
            i = 10
            return i + 15
        self.node_data[key, data_type] = value

    def mark_known_boolean_test(self, node: ast.expr, *, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        For boolean tests that can be determined during decl-visit, we note the AST nodes\n        and the boolean value. This helps us avoid visiting dead code in later passes.\n        '
        self.set_node_data(node, KnownBoolean, KnownBoolean.TRUE if value else KnownBoolean.FALSE)