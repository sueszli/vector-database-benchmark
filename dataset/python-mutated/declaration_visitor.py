from __future__ import annotations
import ast
import sys
from ast import AnnAssign, Assign, AST, AsyncFor, AsyncFunctionDef, AsyncWith, ClassDef, For, FunctionDef, If, Import, ImportFrom, Name, Try, While, With
from typing import List, TYPE_CHECKING, Union
from .module_table import DeferredValue, ModuleTable
from .types import AwaitableTypeRef, Class, DecoratedMethod, Function, InitSubclassFunction, ModuleInstance, ResolvedTypeRef, TypeEnvironment, TypeName, TypeRef, UnknownDecoratedMethod
from .util import sys_hexversion_check
from .visitor import GenericVisitor
if TYPE_CHECKING:
    from .compiler import Compiler

class NestedScope:

    def declare_class(self, node: AST, klass: Class) -> None:
        if False:
            print('Hello World!')
        pass

    def declare_function(self, func: Function | DecoratedMethod) -> None:
        if False:
            return 10
        pass

    def declare_variable(self, node: AnnAssign, module: ModuleTable) -> None:
        if False:
            return 10
        pass

    def declare_variables(self, node: Assign, module: ModuleTable) -> None:
        if False:
            print('Hello World!')
        pass
TScopeTypes = Union[ModuleTable, Class, Function, NestedScope]

class DeclarationVisitor(GenericVisitor[None]):

    def __init__(self, mod_name: str, filename: str, symbols: Compiler, optimize: int) -> None:
        if False:
            return 10
        module = symbols[mod_name] = ModuleTable(mod_name, filename, symbols)
        super().__init__(module)
        self.scopes: List[TScopeTypes] = [self.module]
        self.optimize = optimize
        self.compiler = symbols
        self.type_env: TypeEnvironment = symbols.type_env

    def finish_bind(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.module.finish_bind()

    def parent_scope(self) -> TScopeTypes:
        if False:
            i = 10
            return i + 15
        return self.scopes[-1]

    def enter_scope(self, scope: TScopeTypes) -> None:
        if False:
            while True:
                i = 10
        self.scopes.append(scope)

    def exit_scope(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.scopes.pop()

    def visitAnnAssign(self, node: AnnAssign) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.parent_scope().declare_variable(node, self.module)

    def visitAssign(self, node: Assign) -> None:
        if False:
            return 10
        self.parent_scope().declare_variables(node, self.module)

    def visitClassDef(self, node: ClassDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        bases = [self.module.resolve_type(base) or self.type_env.dynamic for base in node.bases]
        if not bases:
            bases.append(self.type_env.object)
        with self.compiler.error_sink.error_context(self.filename, node):
            klasses = []
            for base in bases:
                klasses.append(base.make_subclass(TypeName(self.module_name, node.name), bases))
            for cur_type in klasses:
                if type(cur_type) != type(klasses[0]):
                    self.syntax_error('Incompatible subtypes', node)
            klass = klasses[0]
        for base in bases:
            if base is self.type_env.named_tuple:
                klass = self.type_env.dynamic
                break
            if base is self.type_env.protocol:
                klass = self.type_env.dynamic
                break
            if base is self.type_env.typed_dict:
                klass = self.type_env.dynamic
                break
            if base.is_final:
                self.syntax_error(f'Class `{klass.instance.name}` cannot subclass a Final class: `{base.instance.name}`', node)
        parent_scope = self.parent_scope()
        if not isinstance(parent_scope, ModuleTable):
            klass = self.type_env.dynamic
        self.enter_scope(NestedScope() if klass is self.type_env.dynamic else klass)
        for item in node.body:
            with self.compiler.error_sink.error_context(self.filename, item):
                self.visit(item)
        self.exit_scope()
        for d in reversed(node.decorator_list):
            if klass is self.type_env.dynamic:
                break
            with self.compiler.error_sink.error_context(self.filename, d):
                decorator = self.module.resolve_decorator(d) or self.type_env.dynamic
                klass = decorator.resolve_decorate_class(klass, d, self)
        parent_scope.declare_class(node, klass.exact_type())
        self.module.types[node] = klass.exact_type()

    def _visitFunc(self, node: Union[FunctionDef, AsyncFunctionDef]) -> None:
        if False:
            print('Hello World!')
        function = self._make_function(node)
        self.parent_scope().declare_function(function)

    def _make_function(self, node: Union[FunctionDef, AsyncFunctionDef]) -> Function:
        if False:
            print('Hello World!')
        if node.name == '__init_subclass__':
            func = InitSubclassFunction(node, self.module, self.type_ref(node))
            parent_scope = self.parent_scope()
            if isinstance(parent_scope, Class):
                parent_scope.has_init_subclass = True
        else:
            func = Function(node, self.module, self.type_ref(node))
        self.enter_scope(func)
        for item in node.body:
            self.visit(item)
        self.exit_scope()
        func_type = func
        if node.decorator_list:
            func_type = UnknownDecoratedMethod(func)
        self.module.types[node] = func_type
        return func

    def visitFunctionDef(self, node: FunctionDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._visitFunc(node)

    def visitAsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        if False:
            i = 10
            return i + 15
        self._visitFunc(node)

    def type_ref(self, node: Union[FunctionDef, AsyncFunctionDef]) -> TypeRef:
        if False:
            i = 10
            return i + 15
        ann = node.returns
        if not ann:
            res = ResolvedTypeRef(self.type_env.dynamic)
        else:
            res = TypeRef(self.module, ann)
        if isinstance(node, AsyncFunctionDef):
            res = AwaitableTypeRef(res, self.module.compiler)
        return res

    def visitImport(self, node: Import) -> None:
        if False:
            for i in range(10):
                print('nop')
        for name in node.names:
            self.compiler.import_module(name.name, self.optimize)
            asname = name.asname
            if asname is None:
                top_level_module = name.name.split('.')[0]
                self.module.declare_import(top_level_module, None, ModuleInstance(top_level_module, self.compiler))
            else:
                self.module.declare_import(asname, None, ModuleInstance(name.name, self.compiler))

    def visitImportFrom(self, node: ImportFrom) -> None:
        if False:
            while True:
                i = 10
        mod_name = node.module
        if not mod_name or node.level:
            raise NotImplementedError("relative imports aren't supported")
        self.compiler.import_module(mod_name, self.optimize)
        mod = self.compiler.modules.get(mod_name)
        for name in node.names:
            child_name = name.asname or name.name
            if mod is None:
                self.module.declare_import(child_name, None, self.type_env.DYNAMIC)
                continue
            val = mod.get_child(name.name)
            if val is not None:
                self.module.declare_import(child_name, (mod_name, name.name), val)
            else:
                module_as_attribute = f'{mod_name}.{name.name}'
                self.compiler.import_module(module_as_attribute, self.optimize)
                if module_as_attribute in self.compiler.modules:
                    typ = ModuleInstance(module_as_attribute, self.compiler)
                else:
                    typ = DeferredValue(mod_name, name.name, self.compiler)
                self.module.declare_import(child_name, (mod_name, name.name), typ)

    def visitFor(self, node: For) -> None:
        if False:
            i = 10
            return i + 15
        self.enter_scope(NestedScope())
        self.generic_visit(node)
        self.exit_scope()

    def visitAsyncFor(self, node: AsyncFor) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.enter_scope(NestedScope())
        self.generic_visit(node)
        self.exit_scope()

    def visitWhile(self, node: While) -> None:
        if False:
            i = 10
            return i + 15
        self.enter_scope(NestedScope())
        self.generic_visit(node)
        self.exit_scope()

    def visitIf(self, node: If) -> None:
        if False:
            print('Hello World!')
        test = node.test
        if isinstance(test, Name) and test.id == 'TYPE_CHECKING':
            self.visit(node.body)
        else:
            result = sys_hexversion_check(node)
            if result is not None:
                self.module.mark_known_boolean_test(test, value=bool(result))
                if result:
                    self.visit(node.body)
                else:
                    self.visit(node.orelse)
                return
            else:
                self.enter_scope(NestedScope())
                self.visit(node.body)
                self.exit_scope()
        if node.orelse:
            self.enter_scope(NestedScope())
            self.visit(node.orelse)
            self.exit_scope()

    def visitWith(self, node: With) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.enter_scope(NestedScope())
        self.generic_visit(node)
        self.exit_scope()

    def visitAsyncWith(self, node: AsyncWith) -> None:
        if False:
            i = 10
            return i + 15
        self.enter_scope(NestedScope())
        self.generic_visit(node)
        self.exit_scope()

    def visitTry(self, node: Try) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.enter_scope(NestedScope())
        self.generic_visit(node)
        self.exit_scope()