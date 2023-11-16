from __future__ import annotations
import ast
from ast import And, AnnAssign, Assign, AST, AsyncFor, AsyncFunctionDef, Attribute, AugAssign, Await, BinOp, BoolOp, Call, ClassDef, Compare, Constant, DictComp, expr, For, FormattedValue, FunctionDef, GeneratorExp, If, IfExp, Import, ImportFrom, Is, IsNot, JoinedStr, Lambda, ListComp, Module, Name, NameConstant, Return, SetComp, Slice, Starred, Subscript, Try, UnaryOp, While, Yield, YieldFrom
from contextlib import contextmanager
from enum import IntEnum
from typing import cast, Dict, Generator, List, Optional, Sequence, Set, Type, TYPE_CHECKING, Union
from ..consts import SC_CELL, SC_FREE, SC_GLOBAL_EXPLICIT, SC_GLOBAL_IMPLICIT, SC_LOCAL
from ..errors import CollectingErrorSink, TypedSyntaxError
from ..symbols import SymbolVisitor
from .declaration_visitor import GenericVisitor
from .effects import NarrowingEffect, NO_EFFECT, TypeState
from .module_table import ModuleFlag, ModuleTable
from .types import _TMP_VAR_PREFIX, access_path, BoolClass, Callable, CheckedDictInstance, CheckedListInstance, CInstance, Class, ClassVar, CType, Dataclass, EnumType, FinalClass, Function, FunctionContainer, GenericClass, InitVar, IsInstanceEffect, KnownBoolean, MethodType, ModuleInstance, Object, OptionalInstance, resolve_assign_error_msg, resolve_instance_attr_by_name, Slot, TransientDecoratedMethod, TransparentDecoratedMethod, TType, TypeDescr, TypeEnvironment, UnionInstance, Value
if TYPE_CHECKING:
    from .compiler import Compiler

class PreserveRefinedFields:
    pass

class UsedRefinementField:

    def __init__(self, name: str, is_source: bool, is_used: bool) -> None:
        if False:
            return 10
        self.name = name
        self.is_source = is_source
        self.is_used = is_used

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'UsedRefinementField(name={self.name}, is_source={self.is_source}, is_used={self.is_used})'
PRESERVE_REFINED_FIELDS = PreserveRefinedFields()

class BindingScope:

    def __init__(self, node: AST, type_env: TypeEnvironment) -> None:
        if False:
            while True:
                i = 10
        self.node = node
        self.type_state = TypeState()
        self.decl_types: Dict[str, TypeDeclaration] = {}
        self.type_env: TypeEnvironment = type_env

    def branch(self) -> LocalsBranch:
        if False:
            return 10
        return LocalsBranch(self)

    def declare(self, name: str, typ: Value, is_final: bool=False, is_inferred: bool=False) -> TypeDeclaration:
        if False:
            for i in range(10):
                print('nop')
        decl = TypeDeclaration(self.type_env.DYNAMIC if is_inferred else typ, is_final)
        self.decl_types[name] = decl
        self.type_state.local_types[name] = typ
        return decl

class EnumBindingScope(BindingScope):

    def __init__(self, node: AST, type_env: TypeEnvironment, enum_type: EnumType) -> None:
        if False:
            print('Hello World!')
        super().__init__(node, type_env)
        self.enum_type = enum_type

    def declare(self, name: str, typ: Value, is_final: bool=False, is_inferred: bool=False) -> TypeDeclaration:
        if False:
            return 10
        self.enum_type.bind_enum_value(name, typ)
        return super().declare(name, typ, is_final, is_inferred)

class ModuleBindingScope(BindingScope):

    def __init__(self, node: ast.Module, module: ModuleTable, type_env: TypeEnvironment) -> None:
        if False:
            return 10
        super().__init__(node, type_env)
        self.module = module

    def declare(self, name: str, typ: Value, is_final: bool=False, is_inferred: bool=False) -> TypeDeclaration:
        if False:
            i = 10
            return i + 15
        if is_inferred:
            typ = typ.nonliteral().inexact()
            is_inferred = False
        return super().declare(name, typ, is_final=is_final, is_inferred=is_inferred)

class LocalsBranch:
    """Handles branching and merging local variable types"""

    def __init__(self, scope: BindingScope) -> None:
        if False:
            while True:
                i = 10
        self.scope = scope
        self.type_env: TypeEnvironment = scope.type_env
        self.entry_type_state: TypeState = scope.type_state.copy()

    def copy(self) -> TypeState:
        if False:
            for i in range(10):
                print('nop')
        'Make a copy of the current local state'
        return self.scope.type_state.copy()

    def restore(self, state: Optional[TypeState]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Restore the locals to the state when we entered'
        self.scope.type_state = state or self.entry_type_state

    def merge(self, entry_type_state: Optional[TypeState]=None) -> None:
        if False:
            return 10
        'Merge the entry type state, or a specific copy, into the current type state'
        if entry_type_state is None:
            entry_type_state = self.entry_type_state
        local_types = self.scope.type_state.local_types
        refined_fields = self.scope.type_state.refined_fields
        keys_to_remove = []
        for (key, value) in local_types.items():
            if key in entry_type_state.local_types:
                if value != entry_type_state.local_types[key]:
                    local_types[key] = self._join(value, entry_type_state.local_types[key])
            else:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del local_types[key]
        keys_to_remove = [key for key in refined_fields if key not in entry_type_state.refined_fields]
        for key in keys_to_remove:
            del refined_fields[key]
        for key in refined_fields:
            entry_refinement_dict = entry_type_state.refined_fields[key]
            refinement_dict = refined_fields[key]
            keys_to_remove = [key for key in refinement_dict if key not in entry_refinement_dict]
            for key in keys_to_remove:
                del refinement_dict[key]
            for key in refinement_dict:
                (entry_typ, _, entry_nodes) = entry_refinement_dict[key]
                (typ, idx, nodes) = refinement_dict[key]
                refinement_dict[key] = (self._join(entry_typ, typ), idx, entry_nodes | nodes)

    def changed(self) -> bool:
        if False:
            i = 10
            return i + 15
        for key in self.entry_type_state.refined_fields:
            if key in self.scope.type_state.refined_fields and self.scope.type_state.refined_fields[key] != self.entry_type_state.refined_fields[key]:
                return True
        return self.entry_type_state.local_types != self.scope.type_state.local_types

    def _join(self, *types: Value) -> Value:
        if False:
            return 10
        if len(types) == 1:
            return types[0]
        return self.type_env.get_union(tuple((t.klass.inexact_type() for t in types))).instance

class TypeDeclaration:

    def __init__(self, typ: Value, is_final: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.type = typ
        self.is_final = is_final

class TerminalKind(IntEnum):
    NonTerminal = 0
    BreakOrContinue = 1
    RaiseOrReturn = 2

class TypeBinder(GenericVisitor[Optional[NarrowingEffect]]):
    """Walks an AST and produces an optionally strongly typed AST, reporting errors when
    operations are occuring that are not sound.  Strong types are based upon places where
    annotations occur which opt-in the strong typing"""

    def __init__(self, symbols: SymbolVisitor, filename: str, compiler: Compiler, module_name: str, optimize: int, enable_patching: bool=False) -> None:
        if False:
            return 10
        module = compiler[module_name]
        super().__init__(module)
        self.symbols = symbols
        self.scopes: List[BindingScope] = []
        self.modules: Dict[str, ModuleTable] = compiler.modules
        self.optimize = optimize
        self.terminals: Dict[AST, TerminalKind] = {}
        self.type_env: TypeEnvironment = compiler.type_env
        self.inline_depth = 0
        self.inline_calls = 0
        self.enable_patching = enable_patching
        self.current_loop: AST | None = None
        self.loop_may_break: Set[AST] = set()
        self.visiting_assignment_target = False
        self._refined_tmpvar_indices: Dict[str, int] = {}

    @property
    def nodes_default_dynamic(self) -> bool:
        if False:
            return 10
        return not self.error_sink.throwing

    @property
    def type_state(self) -> TypeState:
        if False:
            return 10
        return self.binding_scope.type_state

    @property
    def decl_types(self) -> Dict[str, TypeDeclaration]:
        if False:
            for i in range(10):
                print('nop')
        return self.binding_scope.decl_types

    @property
    def binding_scope(self) -> BindingScope:
        if False:
            while True:
                i = 10
        return self.scopes[-1]

    @property
    def scope(self) -> AST:
        if False:
            for i in range(10):
                print('nop')
        return self.binding_scope.node

    def maybe_set_local_type(self, name: str, local_type: Value) -> Value:
        if False:
            return 10
        decl = self.get_target_decl(name)
        assert decl is not None
        decl_type = decl.type
        if local_type is self.type_env.DYNAMIC or not decl_type.klass.can_be_narrowed:
            local_type = decl_type
        self.type_state.local_types[name] = local_type
        return local_type

    def maybe_get_current_class(self) -> Optional[Class]:
        if False:
            while True:
                i = 10
        node = self.scope
        if isinstance(node, ClassDef):
            res = self.get_type(node)
            assert isinstance(res, Class)
            return res

    def maybe_get_current_enclosing_class(self) -> Optional[Class]:
        if False:
            for i in range(10):
                print('nop')
        for scope in reversed(self.scopes):
            node = scope.node
            if isinstance(node, ClassDef):
                res = self.get_type(node)
                return res if isinstance(res, Class) else None

    def visit(self, node: Union[AST, Sequence[AST]], *args: object) -> Optional[NarrowingEffect]:
        if False:
            while True:
                i = 10
        'This override is only here to give Pyre the return type information.'
        ret = super().visit(node, *args)
        if len(self.scopes) > 0 and isinstance(node, AST) and (not self.get_opt_node_data(node, PreserveRefinedFields)):
            self.type_state.refined_fields.clear()
        if ret is not None:
            assert isinstance(ret, NarrowingEffect)
            return ret
        return None

    def get_final_literal(self, node: AST) -> Optional[ast.Constant]:
        if False:
            print('Hello World!')
        return self.module.get_final_literal(node, self.symbols.scopes[self.scope])

    def declare_local(self, name: str, typ: Value, is_final: bool=False, is_inferred: bool=False) -> None:
        if False:
            while True:
                i = 10
        if name in self.decl_types:
            raise TypedSyntaxError(f'Cannot redefine local variable {name}')
        if isinstance(typ, CInstance):
            self.check_primitive_scope(name)
        self.binding_scope.declare(name, typ, is_final=is_final, is_inferred=is_inferred)

    def check_static_import_flags(self, node: Module) -> None:
        if False:
            i = 10
            return i + 15
        saw_doc_str = False
        for stmt in node.body:
            if isinstance(stmt, ast.Expr):
                val = stmt.value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    if saw_doc_str:
                        break
                    saw_doc_str = True
                else:
                    break
            elif isinstance(stmt, ast.Import):
                continue
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module == '__static__.compiler_flags':
                    for name in stmt.names:
                        if name.name == 'checked_dicts':
                            self.module.flags.add(ModuleFlag.CHECKED_DICTS)
                        elif name.name == 'checked_lists':
                            self.module.flags.add(ModuleFlag.CHECKED_LISTS)

    def visitModule(self, node: Module) -> None:
        if False:
            while True:
                i = 10
        self.scopes.append(ModuleBindingScope(node, self.module, type_env=self.type_env))
        self.check_static_import_flags(node)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    def set_param(self, arg: ast.arg, arg_type: Value, scope: BindingScope) -> None:
        if False:
            for i in range(10):
                print('nop')
        scope.declare(arg.arg, arg_type)
        self.set_type(arg, arg_type)

    def _visitParameters(self, args: ast.arguments, scope: BindingScope) -> None:
        if False:
            print('Hello World!')
        default_index = len(args.defaults or []) - (len(args.posonlyargs) + len(args.args))
        for arg in args.posonlyargs:
            ann = arg.annotation
            if ann:
                self.visitExpectedType(ann, self.type_env.DYNAMIC, 'argument annotation cannot be a primitive')
                arg_type = self.module.resolve_annotation(ann) or self.type_env.dynamic
            elif arg.arg in scope.decl_types:
                default_index += 1
                continue
            else:
                self.perf_warning(f"Missing type annotation for positional-only argument '{arg.arg}' prevents type specialization in Static Python", arg)
                arg_type = self.type_env.dynamic
            if default_index >= 0:
                self.visit(args.defaults[default_index], arg_type.instance)
                self.check_can_assign_from(arg_type, self.get_type(args.defaults[default_index]).klass, args.defaults[default_index])
            default_index += 1
            self.set_param(arg, arg_type.instance, scope)
        for arg in args.args:
            ann = arg.annotation
            if ann:
                self.visitExpectedType(ann, self.type_env.DYNAMIC, 'argument annotation cannot be a primitive')
                arg_type = self.module.resolve_annotation(ann) or self.type_env.dynamic
            elif arg.arg in scope.decl_types:
                default_index += 1
                continue
            else:
                self.perf_warning(f"Missing type annotation for argument '{arg.arg}' prevents type specialization in Static Python", arg)
                arg_type = self.type_env.dynamic
            if default_index >= 0:
                self.visit(args.defaults[default_index], arg_type.instance)
                self.check_can_assign_from(arg_type, self.get_type(args.defaults[default_index]).klass, args.defaults[default_index])
            default_index += 1
            self.set_param(arg, arg_type.instance, scope)
        vararg = args.vararg
        if vararg:
            ann = vararg.annotation
            if ann:
                self.visitExpectedType(ann, self.type_env.DYNAMIC, 'argument annotation cannot be a primitive')
            self.set_param(vararg, self.type_env.tuple.exact_type().instance, scope)
        default_index = len(args.kw_defaults or []) - len(args.kwonlyargs)
        for arg in args.kwonlyargs:
            ann = arg.annotation
            if ann:
                self.visitExpectedType(ann, self.type_env.DYNAMIC, 'argument annotation cannot be a primitive')
                arg_type = self.module.resolve_annotation(ann) or self.type_env.dynamic
            else:
                self.perf_warning(f"Missing type annotation for keyword-only argument '{arg.arg}' prevents type specialization in Static Python", arg)
                arg_type = self.type_env.dynamic
            if default_index >= 0:
                default = args.kw_defaults[default_index]
                if default is not None:
                    self.visit(default, arg_type.instance)
                    self.check_can_assign_from(arg_type, self.get_type(default).klass, default)
            default_index += 1
            self.set_param(arg, arg_type.instance, scope)
        kwarg = args.kwarg
        if kwarg:
            ann = kwarg.annotation
            if ann:
                self.visitExpectedType(ann, self.type_env.DYNAMIC, 'argument annotation cannot be a primitive')
            self.set_param(kwarg, self.type_env.dict.exact_type().instance, scope)

    def new_scope(self, node: AST) -> BindingScope:
        if False:
            while True:
                i = 10
        return BindingScope(node, type_env=self.type_env)

    def get_func_container(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionContainer:
        if False:
            return 10
        function = self.get_type(node)
        if not isinstance(function, FunctionContainer):
            raise RuntimeError('bad value for function')
        return function

    def _visitFunc(self, node: Union[FunctionDef, AsyncFunctionDef]) -> None:
        if False:
            i = 10
            return i + 15
        func = self.get_func_container(node)
        func.bind_function(node, self)
        typ = self.get_type(node)
        if not isinstance(typ, TransientDecoratedMethod):
            if isinstance(self.scope, (FunctionDef, AsyncFunctionDef)):
                typ = self.type_env.DYNAMIC
            self.declare_local(node.name, typ)

    def visitFunctionDef(self, node: FunctionDef) -> None:
        if False:
            return 10
        self._visitFunc(node)

    def visitAsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        if False:
            return 10
        self._visitFunc(node)

    def visitClassDef(self, node: ClassDef) -> None:
        if False:
            for i in range(10):
                print('nop')
        for decorator in node.decorator_list:
            self.visitExpectedType(decorator, self.type_env.DYNAMIC, 'decorator cannot be a primitive')
        for kwarg in node.keywords:
            self.visitExpectedType(kwarg.value, self.type_env.DYNAMIC, 'class kwarg cannot be a primitive')
        is_protocol = False
        for base in node.bases:
            self.visitExpectedType(base, self.type_env.DYNAMIC, 'class base cannot be a primitive')
            base_type = self.get_type(base)
            is_protocol |= base_type is self.type_env.protocol
        res = self.get_type(node)
        if is_protocol:
            self.module.compile_non_static.add(node)
        else:
            if isinstance(res, EnumType):
                scope = EnumBindingScope(node, self.type_env, res)
            else:
                scope = BindingScope(node, self.type_env)
            self.scopes.append(scope)
            for stmt in node.body:
                self.visit(stmt)
            self.scopes.pop()
        self.declare_local(node.name, res)

    def set_type(self, node: AST, type: Value) -> None:
        if False:
            return 10
        self.module.types[node] = type

    def get_type(self, node: AST) -> Value:
        if False:
            for i in range(10):
                print('nop')
        if self.nodes_default_dynamic:
            return self.module.types.get(node, self.type_env.DYNAMIC)
        assert node in self.module.types, f'node not found: {node}, {node.lineno}'
        return self.module.types[node]

    def get_node_data(self, key: AST, data_type: Type[TType]) -> TType:
        if False:
            i = 10
            return i + 15
        return self.module.get_node_data(key, data_type)

    def get_opt_node_data(self, key: AST, data_type: Type[TType]) -> TType | None:
        if False:
            return 10
        return self.module.get_opt_node_data(key, data_type)

    def set_node_data(self, key: AST, data_type: Type[TType], value: TType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.module.set_node_data(key, data_type, value)

    def check_primitive_scope(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        cur_scope = self.symbols.scopes[self.scope]
        var_scope = cur_scope.check_name(name)
        if var_scope != SC_LOCAL or isinstance(self.scope, Module):
            raise TypedSyntaxError('cannot use primitives in global or closure scope')

    def get_var_scope(self, var_id: str) -> Optional[int]:
        if False:
            while True:
                i = 10
        cur_scope = self.symbols.scopes[self.scope]
        var_scope = cur_scope.check_name(var_id)
        return var_scope

    def _check_final_attribute_reassigned(self, target: AST, assignment: Optional[AST]) -> None:
        if False:
            for i in range(10):
                print('nop')
        member = None
        klass = None
        member_name = None
        scope = self.scope
        if isinstance(target, ast.Name) and isinstance(scope, ast.ClassDef):
            klass = self.maybe_get_current_class()
            assert isinstance(klass, Class)
            member_name = target.id
            member = klass.get_member(member_name)
        elif isinstance(target, ast.Attribute):
            val = self.get_type(target.value)
            member_name = target.attr
            if isinstance(val, Class):
                klass = val
            else:
                klass = val.klass
            member = klass.get_member(member_name)
        if klass is not None and member is not None and (isinstance(member, Slot) and member.is_final and (member.assignment != assignment) or (isinstance(member, Function) and member.is_final) or (isinstance(member, TransparentDecoratedMethod) and isinstance(member.function, Function) and member.function.is_final)):
            self.syntax_error(f'Cannot assign to a Final attribute of {klass.instance.name}:{member_name}', target)

    def visitAnnAssign(self, node: AnnAssign) -> None:
        if False:
            i = 10
            return i + 15
        self.visitExpectedType(node.annotation, self.type_env.DYNAMIC, 'annotation can not be a primitive value')
        target = node.target
        comp_type = self.module.resolve_annotation(node.annotation, is_declaration=True) or self.type_env.dynamic
        is_final = False
        (comp_type, wrapper) = (comp_type.unwrap(), type(comp_type))
        if wrapper in (ClassVar, InitVar) and (not isinstance(self.scope, ClassDef)):
            self.syntax_error(f'{wrapper.__name__} is allowed only in class attribute annotations.', node)
        if wrapper is FinalClass:
            is_final = True
        declared_type = comp_type.instance
        is_dynamic_final = is_final and declared_type is self.type_env.DYNAMIC
        if isinstance(target, Name):
            if is_dynamic_final:
                value = node.value
                if value:
                    self.visit(value)
                    declared_type = self.get_type(value)
            self.declare_local(target.id, declared_type, is_final)
            self.set_type(target, declared_type)
        with self.in_target():
            self.visit(target)
        value = node.value
        if isinstance(self.scope, ClassDef):
            scope_type = self.get_type(self.scope)
            if isinstance(scope_type, Dataclass) and isinstance(target, Name):
                value = scope_type.bind_field(target.id, value, self)
        if value and (not is_dynamic_final):
            self.visitExpectedType(value, declared_type)
            if isinstance(target, Name):
                new_type = self.get_type(value)
                local_type = self.maybe_set_local_type(target.id, new_type)
                self.set_type(target, local_type)
            self._check_final_attribute_reassigned(target, node)

    def visitAugAssign(self, node: AugAssign) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.target)
        target_type = self.get_type(node.target).inexact()
        self.visit(node.value, target_type)
        self.set_type(node, target_type)

    @contextmanager
    def in_target(self) -> Generator[None, None, None]:
        if False:
            print('Hello World!')
        prev = self.visiting_assignment_target
        self.visiting_assignment_target = True
        try:
            yield
        finally:
            self.visiting_assignment_target = prev

    def visitNamedExpr(self, node: ast.NamedExpr, type_ctx: Optional[Class]=None) -> None:
        if False:
            print('Hello World!')
        target = node.target
        with self.in_target():
            self.visit(target)
        target_type = self.get_type(target)
        self.visit(node.value, target_type)
        value_type = self.get_type(node.value)
        self.assign_value(target, value_type)
        self.set_type(node, self.get_type(target))

    def visitAssign(self, node: Assign) -> None:
        if False:
            i = 10
            return i + 15
        narrowest_target_type = None
        for target in reversed(node.targets):
            cur_type = None
            if isinstance(target, ast.Name):
                decl_type = self.get_target_decl(target.id)
                if decl_type is not None:
                    cur_type = decl_type.type
            elif isinstance(target, (ast.Tuple, ast.List)):
                with self.in_target():
                    self.visit(target)
            else:
                self.visit(target)
                cur_type = self.get_type(target)
            if cur_type is not None and (narrowest_target_type is None or narrowest_target_type.klass.can_assign_from(cur_type.klass)):
                narrowest_target_type = cur_type
        self.visit(node.value, narrowest_target_type)
        value_type = self.get_type(node.value)
        for target in reversed(node.targets):
            self.assign_value(target, value_type, src=node.value, assignment=node)
        if len(node.targets) == 1 and self.is_refinable(node.targets[0]):
            self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and narrowest_target_type != value_type and self.type_env.dynamic.can_assign_from(value_type.klass):
                assert isinstance(target.value, ast.Name)
                self.type_state.refined_fields.setdefault(target.value.id, {})[target.attr] = (value_type, self.refined_field_index(access_path(target)), {target})
        self.set_type(node, value_type)

    def visitPass(self, node: ast.Pass) -> None:
        if False:
            print('Hello World!')
        self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)

    def check_can_assign_from(self, dest: Class, src: Class, node: AST, reason: str='type mismatch: {} cannot be assigned to {}') -> None:
        if False:
            i = 10
            return i + 15
        if not dest.can_assign_from(src) and (src is not self.type_env.dynamic or isinstance(dest, CType)):
            reason = resolve_assign_error_msg(dest, src, reason)
            self.syntax_error(reason, node)

    def clear_refinements_for_nonbool_test(self, test_node: ast.AST) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.get_type(test_node).klass is not self.type_env.bool:
            self.type_state.refined_fields.clear()

    def visitAssert(self, node: ast.Assert) -> None:
        if False:
            print('Hello World!')
        effect = self.visit(node.test) or NO_EFFECT
        effect.apply(self.type_state)
        self.clear_refinements_for_nonbool_test(node.test)
        self.set_node_data(node, NarrowingEffect, effect)
        self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
        message = node.msg
        if message:
            self.visitExpectedType(message, self.type_env.DYNAMIC, 'assert message cannot be a primitive')

    def visitBoolOp(self, node: BoolOp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        effect = NO_EFFECT
        final_type = None
        if isinstance(node.op, And):
            for value in node.values:
                new_effect = self.visit(value) or NO_EFFECT
                effect = effect.and_(new_effect)
                final_type = self.widen(final_type, self.get_type(value))
                new_effect.apply(self.type_state)
            effect.undo(self.type_state)
        elif isinstance(node.op, ast.Or):
            for value in node.values[:-1]:
                new_effect = self.visit(value) or NO_EFFECT
                effect = effect.or_(new_effect)
                old_type = self.get_type(value)
                new_effect.apply(self.type_state)
                self.visit(value)
                new_effect.undo(self.type_state)
                final_type = self.widen(final_type, old_type.klass.opt_type.instance if isinstance(old_type, OptionalInstance) else old_type)
                self.set_type(value, old_type)
                new_effect.reverse(self.type_state)
            new_effect = self.visit(node.values[-1]) or NO_EFFECT
            final_type = self.widen(final_type, self.get_type(node.values[-1]))
            effect.undo(self.type_state)
            effect = effect.or_(new_effect)
        else:
            for value in node.values:
                self.visit(value)
                final_type = self.widen(final_type, self.get_type(value))
        self.set_type(node, final_type or self.type_env.DYNAMIC)
        return effect

    def visitBinOp(self, node: BinOp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            i = 10
            return i + 15
        if isinstance(node.op, ast.Pow):
            type_ctx = None
        self.visit(node.left, type_ctx)
        self.visit(node.right, type_ctx)
        ltype = self.get_type(node.left)
        rtype = self.get_type(node.right)
        tried_right = False
        if ltype.klass.exact_type() in rtype.klass.mro[1:]:
            if rtype.bind_reverse_binop(node, self, type_ctx):
                return NO_EFFECT
            tried_right = True
        if ltype.bind_binop(node, self, type_ctx):
            return NO_EFFECT
        if not tried_right:
            rtype.bind_reverse_binop(node, self, type_ctx)
        return NO_EFFECT

    def visitUnaryOp(self, node: UnaryOp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        effect = self.visit(node.operand, type_ctx)
        self.get_type(node.operand).bind_unaryop(node, self, type_ctx)
        if effect is not None and effect is not NO_EFFECT and isinstance(node.op, ast.Not):
            return effect.not_()
        return NO_EFFECT

    def visitLambda(self, node: Lambda, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            i = 10
            return i + 15
        scope = BindingScope(node, type_env=self.type_env)
        self._visitParameters(node.args, scope)
        self.scopes.append(scope)
        self.visitExpectedType(node.body, self.type_env.DYNAMIC, 'lambda cannot return primitive value')
        self.scopes.pop()
        self.set_type(node, self.type_env.DYNAMIC)
        return NO_EFFECT

    def visitIfExp(self, node: IfExp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        effect = self.visit(node.test) or NO_EFFECT
        effect.apply(self.type_state)
        self.clear_refinements_for_nonbool_test(node.test)
        self.visit(node.body, type_ctx)
        effect.reverse(self.type_state)
        self.visit(node.orelse, type_ctx)
        effect.undo(self.type_state)
        body_t = self.get_type(node.body)
        else_t = self.get_type(node.orelse)
        self.set_type(node, self.type_env.get_union((body_t.klass, else_t.klass)).instance)
        return NO_EFFECT

    def visitSlice(self, node: Slice, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            return 10
        lower = node.lower
        if lower:
            self.visitExpectedType(lower, self.type_env.DYNAMIC, 'slice indices cannot be primitives')
        upper = node.upper
        if upper:
            self.visitExpectedType(upper, self.type_env.DYNAMIC, 'slice indices cannot be primitives')
        step = node.step
        if step:
            self.visitExpectedType(step, self.type_env.DYNAMIC, 'slice indices cannot be primitives')
        self.set_type(node, self.type_env.slice.instance)
        return NO_EFFECT

    def widen(self, existing: Optional[Value], new: Value) -> Value:
        if False:
            return 10
        if existing is None or new.klass.can_assign_from(existing.klass):
            return new
        elif existing.klass.can_assign_from(new.klass):
            return existing
        return self.type_env.get_union((existing.klass, new.klass)).instance

    def visitDict(self, node: ast.Dict, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            i = 10
            return i + 15
        key_type: Optional[Value] = None
        value_type: Optional[Value] = None
        for (k, v) in zip(node.keys, node.values):
            if k:
                self.visitExpectedType(k, self.type_env.DYNAMIC, 'dict keys cannot be primitives')
                key_type = self.widen(key_type, self.get_type(k))
                self.visitExpectedType(v, self.type_env.DYNAMIC, 'dict keys cannot be primitives')
                value_type = self.widen(value_type, self.get_type(v))
            else:
                self.visitExpectedType(v, self.type_env.DYNAMIC, 'dict splat cannot be a primitive')
                d_type = self.get_type(v).klass
                if d_type.generic_type_def is self.type_env.checked_dict:
                    assert isinstance(d_type, GenericClass)
                    key_type = self.widen(key_type, d_type.type_args[0].instance)
                    value_type = self.widen(value_type, d_type.type_args[1].instance)
                elif d_type in (self.type_env.dict, self.type_env.dict.exact_type(), self.type_env.dynamic):
                    key_type = self.type_env.DYNAMIC
                    value_type = self.type_env.DYNAMIC
        self.set_dict_type(node, key_type, value_type, type_ctx)
        return NO_EFFECT

    def set_dict_type(self, node: ast.expr, key_type: Optional[Value], value_type: Optional[Value], type_ctx: Optional[Class]) -> Value:
        if False:
            while True:
                i = 10
        if not isinstance(type_ctx, CheckedDictInstance):
            if ModuleFlag.CHECKED_DICTS in self.module.flags and key_type is not None and (value_type is not None):
                typ = self.type_env.get_generic_type(self.type_env.checked_dict, (key_type.klass.inexact_type(), value_type.klass.inexact_type())).instance
            else:
                typ = self.type_env.dict.exact_type().instance
            self.set_type(node, typ)
            return typ
        assert type_ctx is not None
        type_class = type_ctx.klass
        assert type_class.generic_type_def is self.type_env.checked_dict, type_class
        assert isinstance(type_class, GenericClass)
        if key_type is None:
            key_type = type_class.type_args[0].instance
        if value_type is None:
            value_type = type_class.type_args[1].instance
        gen_type = self.type_env.get_generic_type(self.type_env.checked_dict, (key_type.klass, value_type.klass))
        self.set_type(node, type_ctx)
        if not type_class.type_args[0].can_assign_from(key_type.klass) or not type_class.type_args[1].can_assign_from(value_type.klass):
            self.check_can_assign_from(type_class, gen_type, node)
        return type_ctx

    def set_list_type(self, node: ast.expr, item_type: Optional[Value], type_ctx: Optional[Class]) -> Value:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(type_ctx, CheckedListInstance):
            if ModuleFlag.CHECKED_LISTS in self.module.flags and item_type is not None:
                typ = self.type_env.get_generic_type(self.type_env.checked_list, (item_type.nonliteral().klass.inexact_type(),)).instance
            else:
                typ = self.type_env.list.exact_type().instance
            self.set_type(node, typ)
            return typ
        assert type_ctx is not None
        type_class = type_ctx.klass
        assert type_class.generic_type_def is self.type_env.checked_list, type_class
        assert isinstance(type_class, GenericClass)
        if item_type is None:
            item_type = type_class.type_args[0].instance
        gen_type = self.type_env.get_generic_type(self.type_env.checked_list, (item_type.nonliteral().klass.inexact_type(),))
        self.set_type(node, type_ctx)
        if not type_class.type_args[0].can_assign_from(item_type.klass):
            self.check_can_assign_from(type_class, gen_type, node)
        return type_ctx

    def visitSet(self, node: ast.Set, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            print('Hello World!')
        for elt in node.elts:
            self.visitExpectedType(elt, self.type_env.DYNAMIC, 'set members cannot be primitives')
        self.set_type(node, self.type_env.set.exact_type().instance)
        return NO_EFFECT

    def visitGeneratorExp(self, node: GeneratorExp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            print('Hello World!')
        self.visit_comprehension(node, node.generators, node.elt)
        self.set_type(node, self.type_env.DYNAMIC)
        return NO_EFFECT

    def visitListComp(self, node: ListComp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            while True:
                i = 10
        self.visit_comprehension(node, node.generators, node.elt)
        item_type = self.get_type(node.elt)
        self.set_list_type(node, item_type, type_ctx)
        return NO_EFFECT

    def visitSetComp(self, node: SetComp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            while True:
                i = 10
        self.visit_comprehension(node, node.generators, node.elt)
        self.set_type(node, self.type_env.set.exact_type().instance)
        return NO_EFFECT

    def get_target_decl(self, name: str) -> Optional[TypeDeclaration]:
        if False:
            i = 10
            return i + 15
        decl_type = self.decl_types.get(name)
        if decl_type is None:
            scope_type = self.get_var_scope(name)
            if scope_type in (SC_GLOBAL_EXPLICIT, SC_GLOBAL_IMPLICIT):
                decl_type = self.scopes[0].decl_types.get(name)
        return decl_type

    def assign_value(self, target: expr, value: Value, src: Optional[expr]=None, assignment: Optional[AST]=None) -> None:
        if False:
            while True:
                i = 10
        if isinstance(target, Name):
            decl_type = self.get_target_decl(target.id)
            if decl_type is None:
                self.declare_local(target.id, value, is_inferred=True)
            else:
                if decl_type.is_final:
                    self.syntax_error('Cannot assign to a Final variable', target)
                self.check_can_assign_from(decl_type.type.klass, value.klass, target)
            local_type = self.maybe_set_local_type(target.id, value)
            self.set_type(target, local_type)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(src, (ast.Tuple, ast.List)) and len(target.elts) == len(src.elts):
                for (target, inner_value) in zip(target.elts, src.elts):
                    self.assign_value(target, self.get_type(inner_value), src=inner_value)
            elif isinstance(src, ast.Constant):
                t = src.value
                if isinstance(t, tuple) and len(t) == len(target.elts):
                    for (target, inner_value) in zip(target.elts, t):
                        self.assign_value(target, self.type_env.constant_types[type(inner_value)])
                else:
                    for val in target.elts:
                        self.assign_value(val, self.type_env.DYNAMIC)
            else:
                for val in target.elts:
                    self.assign_value(val, self.type_env.DYNAMIC)
        else:
            self.check_can_assign_from(self.get_type(target).klass, value.klass, target)
        self._check_final_attribute_reassigned(target, assignment)

    def visitDictComp(self, node: DictComp, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            while True:
                i = 10
        self.visit(node.generators[0].iter)
        scope = BindingScope(node, type_env=self.type_env)
        self.scopes.append(scope)
        iter_type = self.get_type(node.generators[0].iter).get_iter_type(node.generators[0].iter, self)
        with self.in_target():
            self.visit(node.generators[0].target)
        self.assign_value(node.generators[0].target, iter_type)
        for if_ in node.generators[0].ifs:
            self.visit(if_)
        for gen in node.generators[1:]:
            self.visit(gen.iter)
            iter_type = self.get_type(gen.iter).get_iter_type(gen.iter, self)
            self.assign_value(gen.target, iter_type)
        self.visitExpectedType(node.key, self.type_env.DYNAMIC, 'dictionary comprehension key cannot be a primitive')
        self.visitExpectedType(node.value, self.type_env.DYNAMIC, 'dictionary comprehension value cannot be a primitive')
        self.scopes.pop()
        key_type = self.get_type(node.key)
        value_type = self.get_type(node.value)
        self.set_dict_type(node, key_type, value_type, type_ctx)
        return NO_EFFECT

    def visit_comprehension(self, node: ast.expr, generators: List[ast.comprehension], *elts: ast.expr) -> None:
        if False:
            while True:
                i = 10
        self.visit(generators[0].iter)
        scope = BindingScope(node, type_env=self.type_env)
        self.scopes.append(scope)
        iter_type = self.get_type(generators[0].iter).get_iter_type(generators[0].iter, self)
        with self.in_target():
            self.visit(generators[0].target)
        self.assign_value(generators[0].target, iter_type)
        for if_ in generators[0].ifs:
            self.visit(if_)
        for gen in generators[1:]:
            self.visit(gen.iter)
            iter_type = self.get_type(gen.iter).get_iter_type(gen.iter, self)
            with self.in_target():
                self.visit(gen.target)
            self.assign_value(gen.target, iter_type)
            for if_ in gen.ifs:
                self.visit(if_)
        for elt in elts:
            self.visitExpectedType(elt, self.type_env.DYNAMIC, 'generator element cannot be a primitive')
        self.scopes.pop()

    def visitAwait(self, node: Await, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        self.visitExpectedType(node.value, self.type_env.DYNAMIC, 'cannot await a primitive value')
        self.get_type(node.value).bind_await(node, self, type_ctx)
        return NO_EFFECT

    def visitYield(self, node: Yield, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        value = node.value
        if value is not None:
            self.visitExpectedType(value, self.type_env.DYNAMIC, 'cannot yield a primitive value')
        self.set_type(node, self.type_env.DYNAMIC)
        return NO_EFFECT

    def visitYieldFrom(self, node: YieldFrom, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            return 10
        self.visitExpectedType(node.value, self.type_env.DYNAMIC, 'cannot yield from a primitive value')
        self.set_type(node, self.type_env.DYNAMIC)
        return NO_EFFECT

    def visitCompare(self, node: Compare, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            print('Hello World!')
        if len(node.ops) == 1 and isinstance(node.ops[0], (Is, IsNot)):
            self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
            left = node.left
            right = node.comparators[0]
            other = None
            self.set_type(node, self.type_env.bool.instance)
            self.set_type(node.ops[0], self.type_env.bool.instance)
            self.visit(left)
            self.visit(right)
            if isinstance(left, (Constant, NameConstant)) and left.value is None:
                other = right
            elif isinstance(right, (Constant, NameConstant)) and right.value is None:
                other = left
            if other is not None and self.is_refinable(other):
                var_type = self.get_type(other)
                if isinstance(var_type, UnionInstance) and (not var_type.klass.is_generic_type_definition):
                    assert isinstance(other, (ast.Name, ast.Attribute))
                    effect = IsInstanceEffect(other, var_type, self.type_env.none.instance, self)
                    if isinstance(node.ops[0], IsNot):
                        effect = effect.not_()
                    return effect
        self.visit(node.left)
        left = node.left
        ltype = self.get_type(node.left)
        node.ops = [type(op)() for op in node.ops]
        for (comparator, op) in zip(node.comparators, node.ops):
            self.visit(comparator)
            rtype = self.get_type(comparator)
            tried_right = False
            if ltype.klass.exact_type() in rtype.klass.mro[1:]:
                if ltype.bind_reverse_compare(node, left, op, comparator, self, type_ctx):
                    continue
                tried_right = True
            if ltype.bind_compare(node, left, op, comparator, self, type_ctx):
                continue
            if not tried_right:
                rtype.bind_reverse_compare(node, left, op, comparator, self, type_ctx)
            ltype = rtype
            right = comparator
        return NO_EFFECT

    def visitCall(self, node: Call, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            print('Hello World!')
        self.visit(node.func)
        return self.get_type(node.func).bind_call(node, self, type_ctx)

    def visitFormattedValue(self, node: FormattedValue, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            print('Hello World!')
        self.visitExpectedType(node.value, self.type_env.DYNAMIC, 'cannot use primitive in formatted value')
        if (fs := node.format_spec):
            self.visit(fs)
        self.set_type(node, self.type_env.DYNAMIC)
        return NO_EFFECT

    def visitJoinedStr(self, node: JoinedStr, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            while True:
                i = 10
        for value in node.values:
            self.visit(value)
        self.set_type(node, self.type_env.str.exact_type().instance)
        return NO_EFFECT

    def visitConstant(self, node: Constant, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            return 10
        self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
        if type_ctx is not None:
            type_ctx.bind_constant(node, self)
        else:
            self.type_env.DYNAMIC.bind_constant(node, self)
        return NO_EFFECT

    def visitAttribute(self, node: Attribute, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            i = 10
            return i + 15
        value = node.value
        self.visit(value)
        base = self.get_type(value)
        base.bind_attr(node, self, type_ctx)
        if isinstance(value, ast.Name) and value.id in self.type_state.refined_fields and (node.attr in self.type_state.refined_fields[value.id]):
            if isinstance(node.ctx, ast.Load):
                (typ, idx, source_nodes) = self.type_state.refined_fields[value.id][node.attr]
                self.set_type(node, typ)
                temp_name = self._refined_field_name(idx)
                for source_node in source_nodes:
                    is_used = node != source_node
                    self.set_node_data(source_node, UsedRefinementField, UsedRefinementField(temp_name, True, is_used))
                if node not in source_nodes:
                    self.set_node_data(node, UsedRefinementField, UsedRefinementField(temp_name, False, True))
            elif node.attr in self.type_state.refined_fields[value.id]:
                del self.type_state.refined_fields[value.id][node.attr]
        if isinstance(base, ModuleInstance):
            self.set_node_data(node, TypeDescr, (base.module_name, node.attr))
        if self.is_refinable(node):
            self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
            if isinstance(node.ctx, ast.Store):
                temp_name = self._refined_field_name(self.refined_field_index(access_path(node)))
                self.set_node_data(node, UsedRefinementField, UsedRefinementField(temp_name, True, False))
        return NO_EFFECT

    def visitSubscript(self, node: Subscript, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.value)
        self.visit(node.slice)
        val_type = self.get_type(node.value)
        val_type.bind_subscr(node, self.get_type(node.slice), self, type_ctx)
        return NO_EFFECT

    def visitStarred(self, node: Starred, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            return 10
        self.visitExpectedType(node.value, self.type_env.DYNAMIC, 'cannot use primitive in starred expression')
        self.set_type(node, self.type_env.DYNAMIC)
        return NO_EFFECT

    def visitName(self, node: Name, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            i = 10
            return i + 15
        self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
        found_name = self.visiting_assignment_target
        cur_scope = self.symbols.scopes[self.scope]
        scope = cur_scope.check_name(node.id)
        if scope == SC_LOCAL and (not isinstance(self.scope, Module)):
            if node.id in self.type_state.local_types:
                found_name = True
            var_type = self.type_state.local_types.get(node.id, self.type_env.DYNAMIC)
            self.set_type(node, var_type)
        else:
            (typ, descr) = self.module.resolve_name_with_descr(node.id)
            if typ is None and len(self.scopes) > 0:
                decl = self.scopes[0].decl_types.get(node.id)
                if decl is not None:
                    typ = decl.type
            self.set_type(node, typ or self.type_env.DYNAMIC)
            if descr is not None:
                self.set_node_data(node, TypeDescr, descr)
            if typ is not None:
                found_name = True
        if not found_name:
            if scope == SC_FREE:
                for scope in reversed(self.scopes):
                    if node.id in scope.type_state.local_types:
                        found_name = True
                        break
            elif scope == SC_CELL:
                found_name = node.id in self.scopes[-1].type_state.local_types
        if not found_name:
            raise TypedSyntaxError(f'Name `{node.id}` is not defined.')
        type = self.get_type(node)
        if isinstance(type, UnionInstance) and (not type.klass.is_generic_type_definition):
            effect = IsInstanceEffect(node, type, self.type_env.none.instance, self)
            return effect.not_()
        return NO_EFFECT

    def visitExpectedType(self, node: AST, expected: Value, reason: str='type mismatch: {} cannot be assigned to {}', blame: Optional[AST]=None) -> Optional[NarrowingEffect]:
        if False:
            return 10
        res = self.visit(node, expected)
        self.check_can_assign_from(expected.klass, self.get_type(node).klass, blame or node, reason)
        return res

    def visitList(self, node: ast.List, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            for i in range(10):
                print('nop')
        item_type: Optional[Value] = None
        for elt in node.elts:
            self.visitExpectedType(elt, self.type_env.DYNAMIC)
            if isinstance(elt, ast.Starred):
                unpacked_value_type = self.get_type(elt.value)
                if isinstance(unpacked_value_type, CheckedListInstance):
                    element_type = unpacked_value_type.klass.type_args[0].instance
                else:
                    element_type = self.type_env.DYNAMIC
            else:
                element_type = self.get_type(elt)
            item_type = self.widen(item_type, element_type)
        self.set_list_type(node, item_type, type_ctx)
        return NO_EFFECT

    def visitTuple(self, node: ast.Tuple, type_ctx: Optional[Class]=None) -> NarrowingEffect:
        if False:
            i = 10
            return i + 15
        for elt in node.elts:
            self.visitExpectedType(elt, self.type_env.DYNAMIC)
        self.set_type(node, self.type_env.tuple.exact_type().instance)
        return NO_EFFECT

    def set_terminal_kind(self, node: AST, level: TerminalKind) -> None:
        if False:
            print('Hello World!')
        current = self.terminals.get(node, TerminalKind.NonTerminal)
        if current < level:
            self.terminals[node] = level

    def visitContinue(self, node: ast.Continue) -> None:
        if False:
            while True:
                i = 10
        self.set_node_data(node, AST, self.current_loop)
        self.set_terminal_kind(node, TerminalKind.BreakOrContinue)

    def visitBreak(self, node: ast.Break) -> None:
        if False:
            while True:
                i = 10
        self.set_terminal_kind(node, TerminalKind.BreakOrContinue)
        if self.current_loop is not None:
            self.loop_may_break.add(self.current_loop)

    def visitRaise(self, node: ast.Raise) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.set_terminal_kind(node, TerminalKind.RaiseOrReturn)
        self.generic_visit(node)

    def visitReturn(self, node: Return) -> None:
        if False:
            print('Hello World!')
        self.set_terminal_kind(node, TerminalKind.RaiseOrReturn)
        value = node.value
        if value is not None:
            cur_scope = self.binding_scope
            func = cur_scope.node
            expected = self.type_env.DYNAMIC
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = self.get_func_container(func)
                expected = function.get_expected_return()
            self.visit(value, expected)
            returned = self.get_type(value).klass
            if returned is not self.type_env.dynamic and (not expected.klass.can_assign_from(returned)):
                reason = resolve_assign_error_msg(expected.klass, returned, 'return type must be {1}, not {0}')
                self.syntax_error(reason, node)

    def visitImport(self, node: Import) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(self.scope, (FunctionDef, AsyncFunctionDef)):
            for name in node.names:
                import_name = name.name.split('.')[0] if name.asname is None else name.name
                declaration_name = name.asname or import_name.split('.')[0]
                if import_name in self.compiler.modules:
                    typ = ModuleInstance(import_name, self.compiler)
                else:
                    typ = self.type_env.DYNAMIC
                self.declare_local(declaration_name, typ)

    def visitImportFrom(self, node: ImportFrom) -> None:
        if False:
            while True:
                i = 10
        mod_name = node.module
        if node.level or not mod_name:
            raise NotImplementedError("relative imports aren't supported")
        if mod_name == '__static__':
            for alias in node.names:
                name = alias.name
                if name == '*':
                    self.syntax_error('from __static__ import * is disallowed', node)
                elif self.compiler.statics.get_child(name) is None:
                    self.syntax_error(f'unsupported static import {name}', node)
        if mod_name not in self.compiler.modules:
            for alias in node.names:
                asname = alias.asname
                name: str = asname if asname is not None else alias.name
                self.declare_local(name, self.type_env.DYNAMIC)
        elif isinstance(self.scope, (FunctionDef, AsyncFunctionDef)):
            for alias in node.names:
                asname = alias.asname
                name: str = asname if asname is not None else alias.name
                self.declare_local(name, self.compiler.modules[mod_name].get_child(alias.name, self.type_env.DYNAMIC))

    def visit_check_terminal(self, nodes: Sequence[ast.stmt]) -> TerminalKind:
        if False:
            while True:
                i = 10
        ret = TerminalKind.NonTerminal
        for stmt in nodes:
            self.visit(stmt)
            if ret == TerminalKind.NonTerminal and stmt in self.terminals:
                ret = self.terminals[stmt]
        return ret

    def get_bool_const(self, node: ast.expr) -> bool | None:
        if False:
            return 10
        kb = self.get_opt_node_data(node, KnownBoolean)
        if kb is not None:
            return True if kb == KnownBoolean.TRUE else False

    def visitIf(self, node: If) -> None:
        if False:
            i = 10
            return i + 15
        self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
        effect = self.visit(node.test) or NO_EFFECT
        test_const = self.get_bool_const(node.test)
        visit_body = test_const is not False
        visit_orelse = test_const is not True
        self.clear_refinements_for_nonbool_test(node.test)
        branch = self.binding_scope.branch()
        effect.apply(self.type_state)
        if visit_body:
            terminates = self.visit_check_terminal(node.body)
        else:
            terminates = TerminalKind.NonTerminal
        if visit_orelse and node.orelse:
            if_end = branch.copy()
            branch.restore()
            effect.reverse(self.type_state)
            else_terminates = self.visit_check_terminal(node.orelse)
            if else_terminates:
                if terminates:
                    self.terminals[node] = min(terminates, else_terminates)
                else:
                    branch.restore(if_end)
            elif not terminates:
                branch.merge(if_end)
        elif terminates:
            effect.reverse(self.type_state)
        else:
            branch.merge(effect.reverse(branch.entry_type_state))

    def visitTry(self, node: Try) -> None:
        if False:
            return 10
        branch = self.binding_scope.branch()
        body_terminal = self.visit_check_terminal(node.body)
        post_try = branch.copy()
        branch.merge()
        body_maybe_executed = branch.copy()
        merges = []
        else_terminal = TerminalKind.NonTerminal
        if node.orelse:
            branch.restore(post_try)
            else_terminal = self.visit_check_terminal(node.orelse)
            post_try = branch.copy()
        no_exception_terminal = max(body_terminal, else_terminal)
        terminals = [no_exception_terminal]
        for handler in node.handlers:
            branch.restore(body_maybe_executed.copy())
            self.visit(handler)
            terminal = self.terminals.get(handler, TerminalKind.NonTerminal)
            terminals.append(terminal)
            if terminal == TerminalKind.NonTerminal:
                merges.append(branch.copy())
        branch.restore(post_try)
        for merge in merges:
            branch.merge(merge)
        terminal = min(terminals)
        if node.finalbody:
            finally_terminal = self.visit_check_terminal(node.finalbody)
            if finally_terminal:
                terminal = finally_terminal
        if terminal:
            self.set_terminal_kind(node, terminal)

    def visitExceptHandler(self, node: ast.ExceptHandler) -> None:
        if False:
            for i in range(10):
                print('nop')
        htype = node.type
        hname = None
        if htype:
            self.visit(htype)
            handler_type = self.get_type(htype)
            hname = node.name
            if hname:
                if handler_type is self.type_env.DYNAMIC or not isinstance(handler_type, Class):
                    handler_type = self.type_env.dynamic
                handler_type = handler_type.inexact_type()
                decl_type = self.decl_types.get(hname)
                if decl_type and decl_type.is_final:
                    self.syntax_error('Cannot assign to a Final variable', node)
                self.binding_scope.declare(hname, handler_type.instance)
        terminal = self.visit_check_terminal(node.body)
        if terminal:
            self.set_terminal_kind(node, terminal)
        if hname is not None:
            del self.decl_types[hname]
            del self.type_state.local_types[hname]

    def iterate_to_fixed_point(self, body: Sequence[ast.stmt], test: ast.expr | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Iterate given loop body until local types reach a fixed point.'
        branch: LocalsBranch | None = None
        counter = 0
        entry_decls = self.decl_types.copy()
        while not branch or branch.changed():
            branch = self.binding_scope.branch()
            counter += 1
            if counter > 50:
                raise AssertionError('Too many loops in fixed-point iteration.')
            with self.temporary_error_sink(CollectingErrorSink()):
                if test is not None:
                    effect = self.visit(test) or NO_EFFECT
                    effect.apply(self.type_state)
                    self.clear_refinements_for_nonbool_test(test)
                terminates = self.visit_check_terminal(body)
                self.binding_scope.decl_types = entry_decls.copy()
            branch.merge()

    @contextmanager
    def in_loop(self, node: AST) -> Generator[None, None, None]:
        if False:
            for i in range(10):
                print('nop')
        orig = self.current_loop
        self.current_loop = node
        try:
            yield
        finally:
            self.current_loop = orig

    def visitWhile(self, node: While) -> None:
        if False:
            while True:
                i = 10
        self.set_node_data(node, PreserveRefinedFields, PRESERVE_REFINED_FIELDS)
        branch = self.scopes[-1].branch()
        with self.in_loop(node):
            self.iterate_to_fixed_point(node.body, node.test)
            effect = self.visit(node.test) or NO_EFFECT
            condition_always_true = self.get_type(node.test).is_truthy_literal()
            effect.apply(self.type_state)
            terminal_level = self.visit_check_terminal(node.body)
        self.clear_refinements_for_nonbool_test(node.test)
        does_not_break = node not in self.loop_may_break
        if terminal_level == TerminalKind.RaiseOrReturn and does_not_break:
            branch.restore()
            effect.reverse(self.type_state)
        else:
            branch.merge(effect.reverse(branch.entry_type_state))
        if condition_always_true and does_not_break:
            self.set_terminal_kind(node, terminal_level)
        if node.orelse:
            effect.reverse(self.type_state)
            self.visit(node.orelse)
            branch.merge()

    def visitFor(self, node: For) -> None:
        if False:
            return 10
        with self.in_loop(node):
            self.visit(node.iter)
        container_type = self.get_type(node.iter)
        target_type = container_type.get_iter_type(node.iter, self)
        with self.in_target():
            container_type.bind_forloop_target(node.target, self)
        self.assign_value(node.target, target_type)
        branch = self.scopes[-1].branch()
        with self.in_loop(node):
            self.iterate_to_fixed_point(node.body)
            self.visit(node.body)
        self.visit(node.orelse)
        branch.merge()

    def visitAsyncFor(self, node: AsyncFor) -> None:
        if False:
            return 10
        self.visitExpectedType(node.iter, self.type_env.DYNAMIC, 'cannot await a primitive value')
        target_type = self.type_env.DYNAMIC
        with self.in_target():
            self.visit(node.target)
        self.assign_value(node.target, target_type)
        branch = self.scopes[-1].branch()
        with self.in_loop(node):
            self.iterate_to_fixed_point(node.body)
            self.visit(node.body)
        self.visit(node.orelse)
        branch.merge()

    def visitWith(self, node: ast.With) -> None:
        if False:
            i = 10
            return i + 15
        self.visit(node.items)
        may_suppress_exceptions = False
        for item in node.items:
            expr = item.context_expr
            typ = self.get_type(expr)
            if isinstance(typ, Object):
                exit_method_type = resolve_instance_attr_by_name(expr, '__exit__', typ, self)
                if isinstance(exit_method_type, MethodType):
                    exit_method_type = exit_method_type.function
                if isinstance(exit_method_type, Callable):
                    exit_ret_type = exit_method_type.return_type.resolved()
                    if isinstance(exit_ret_type, BoolClass) and exit_ret_type.literal_value is False:
                        continue
            may_suppress_exceptions = True
        terminates = self.visit_check_terminal(node.body)
        if not may_suppress_exceptions:
            self.set_terminal_kind(node, terminates)

    def visitAsyncWith(self, node: ast.With) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.items)
        for stmt in node.body:
            self.visit(stmt)

    def visitwithitem(self, node: ast.withitem) -> None:
        if False:
            while True:
                i = 10
        self.visit(node.context_expr)
        optional_vars = node.optional_vars
        if optional_vars:
            with self.in_target():
                self.visit(optional_vars)
            self.assign_value(optional_vars, self.type_env.DYNAMIC)

    def is_refinable(self, node: ast.AST) -> bool:
        if False:
            print('Hello World!')
        if isinstance(node, Name):
            return True
        elif isinstance(node, ast.Attribute) and isinstance(node.value, Name):
            typ = self.get_type(node.value)
            slot = typ.klass.find_slot(node)
            if slot:
                return True
        return False

    def refined_field_index(self, access_path: List[str]) -> int:
        if False:
            i = 10
            return i + 15
        key = '.'.join(access_path)
        if key in self._refined_tmpvar_indices:
            return self._refined_tmpvar_indices[key]
        next_index = len(self._refined_tmpvar_indices)
        self._refined_tmpvar_indices[key] = next_index
        return self._refined_tmpvar_indices[key]

    def _refined_field_name(self, idx: int) -> str:
        if False:
            while True:
                i = 10
        return f'{_TMP_VAR_PREFIX}.__refined_field__.{idx}'