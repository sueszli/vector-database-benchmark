from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
import torch
import torch.utils._device
from torch._dynamo.source import is_from_local_source, TensorProperty, TensorPropertySource
from torch._guards import DuplicateInputs, Guard, GuardBuilderBase, GuardEnvExpr, GuardSource, Source
from torch.fx.experimental.symbolic_shapes import EqualityConstraint, is_symbolic, SYMPY_INTERP
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn
from .utils import dict_const_keys, dict_const_keys_repr, dict_param_key_ids, guard_failures, is_guard_failure_reporting_enabled, istype, orig_code_map, tensor_always_has_static_shape, tuple_iterator_getitem, tuple_iterator_len
log = logging.getLogger(__name__)
guards_log = torch._logging.getArtifactLogger(__name__, 'guards')
recompiles_log = torch._logging.getArtifactLogger(__name__, 'recompiles')
verbose_guards_log = torch._logging.getArtifactLogger(__name__, 'verbose_guards')
TensorGuards = torch._C._dynamo.guards.TensorGuards
check_obj_id = torch._C._dynamo.guards.check_obj_id
check_type_id = torch._C._dynamo.guards.check_type_id
dict_version = torch._C._dynamo.guards.dict_version

@functools.lru_cache(None)
def uninteresting_files():
    if False:
        while True:
            i = 10
    import torch._dynamo.external_utils
    mods = [torch._dynamo.external_utils]
    return {inspect.getfile(m) for m in mods}
CLOSURE_VARS = {'___check_type_id': check_type_id, '___check_obj_id': check_obj_id, '___current_backend': lambda : torch._dynamo.eval_frame.guarded_backend_cache.current_backend, '___lookup_backend': lambda backend_obj_id: torch._dynamo.eval_frame.guarded_backend_cache.cached_backends[backend_obj_id], '___skip_backend_check': lambda : torch._dynamo.eval_frame.guarded_backend_cache.skip_backend_check_for_run_only_mode, '___odict_getitem': collections.OrderedDict.__getitem__, '___dict_param_key_ids': dict_param_key_ids, '___dict_const_keys': dict_const_keys, '___dict_version': dict_version, '___dict_contains': lambda a, b: a in b, '___tuple_iterator_len': tuple_iterator_len, '___tuple_iterator_getitem': tuple_iterator_getitem, '__math_isnan': math.isnan, 'inf': float('inf'), '__load_module': importlib.import_module, 'utils_device': torch.utils._device, 'device': torch.device, '___from_numpy': lambda a: torch.as_tensor(a) if isinstance(a, (np.generic, np.ndarray)) else a, 'torch': torch}
if sys.version_info[:2] <= (3, 8):
    try:
        import astunparse

        def _ast_unparse(node: ast.AST) -> str:
            if False:
                while True:
                    i = 10
            return astunparse.unparse(node).replace('\n', '')
        HAS_UNPARSE_FUNCTIONS = True
    except ImportError:
        HAS_UNPARSE_FUNCTIONS = False
        pass
else:
    HAS_UNPARSE_FUNCTIONS = True

    def _ast_unparse(node: ast.AST) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ast.unparse(node).replace('\n', '')

def strip_function_call(name):
    if False:
        while True:
            i = 10
    '\n    "___odict_getitem(a, 1)" => "a"\n    "a.layers[slice(2)][0]._xyz" ==> "a"\n    "getattr(a.layers[slice(2)][0]._abc, \'0\')" ==> "a"\n    "getattr(getattr(a.x[3], \'0\'), \'3\')" ==> "a"\n    "a.layers[slice(None, -1, None)][0]._xyz" ==> "a"\n    '
    valid_name = re.compile('[A-Za-z_].*')
    curr = ''
    for char in name:
        if char in ' (':
            curr = ''
        elif char in '),[]':
            if curr and curr != 'None' and valid_name.match(curr):
                return strip_function_call(curr)
        else:
            curr += char
    return strip_getattr_getitem(name)

def strip_getattr_getitem(name):
    if False:
        i = 10
        return i + 15
    '\n    "a[1]" => "a"\n    "a.foo" => "a"\n    '
    return re.split('[.\\[]', name)[0]

@dataclasses.dataclass
class GuardCodeList:
    code_list: List[str]
    guard: Guard

class GuardBuilder(GuardBuilderBase):

    def __init__(self, id_ref: Callable[[Any], str], source_ref: Callable[[Source], str], lookup_weakrefs: Callable[[Type[object]], ReferenceType[object]], local_scope: Dict[str, object], global_scope: Dict[str, object], check_fn_manager: CheckFunctionManager):
        if False:
            for i in range(10):
                print('nop')
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        self.scope: Dict[str, Dict[str, object]] = {'L': local_scope, 'G': global_scope}
        self.scope['__builtins__'] = builtins.__dict__.copy()
        for (name, package_module) in torch.package.package_importer._package_imported_modules.items():
            name = name.replace('>', '_').replace('<', '_').replace('.', '_dot_')
            self.scope['__builtins__'][name] = package_module
            self.scope[name] = package_module
        self.argnames: List[str] = []
        self.code: List[GuardCodeList] = []
        self.shape_env_code: List[GuardCodeList] = []
        self.tensor_check_names: List[str] = []
        self.tensor_check_examples: List[torch.Tensor] = []
        self.tensor_check_guards: List[Guard] = []
        self.check_fn_manager: CheckFunctionManager = check_fn_manager
        self.id_matched_objs: Dict[str, ReferenceType[object]] = {}

    def get(self, name: str) -> Any:
        if False:
            i = 10
            return i + 15
        return eval(name, self.scope, CLOSURE_VARS)

    def arg_ref(self, guard: Union[str, Guard]) -> str:
        if False:
            for i in range(10):
                print('nop')
        name: str
        if isinstance(guard, str):
            name = guard
        else:
            name = guard.name
        base = strip_getattr_getitem(strip_function_call(name))
        if base not in self.argnames:
            if re.match('[a-zA-Z0-9_]+', base):
                if re.match('^\\d+$', base):
                    log.warning('invalid var name: %s', guard)
                self.argnames.append(base)
        return name

    def TYPE_MATCH(self, guard: Guard):
        if False:
            for i in range(10):
                print('nop')
        t = type(self.get(guard.name))
        obj_id = self.id_ref(t)
        code = f'___check_type_id({self.arg_ref(guard)}, {obj_id})'
        self._produce_guard_code(guard, [code])

    def DICT_VERSION(self, guard: Guard):
        if False:
            return 10
        ref = self.arg_ref(guard)
        version = dict_version(self.get(guard.name))
        code = f'___dict_version({ref}) == {version}'
        self._produce_guard_code(guard, [code])

    def DICT_CONTAINS(self, guard: Guard, key: str, invert: bool):
        if False:
            for i in range(10):
                print('nop')
        dict_ref = self.arg_ref(guard)
        maybe_not = 'not ' if invert else ''
        code = f'{maybe_not}___dict_contains({key!r}, {dict_ref})'
        return self._produce_guard_code(guard, [code])

    def BOOL_FALSE(self, guard: Guard):
        if False:
            print('Hello World!')
        ref = self.arg_ref(guard)
        code = f'not {ref}'
        self._produce_guard_code(guard, [code])

    def ID_MATCH(self, guard: Guard):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(guard.originating_source, TypeSource):
            return self.TYPE_MATCH(Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH))
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        code = f'___check_obj_id({ref}, {self.id_ref(val)})'
        self._produce_guard_code(guard, [code])
        if isinstance(guard.originating_source, LocalSource):
            if isinstance(val, torch.nn.Module):
                local_name = guard.originating_source.local_name
                weak_id = self.lookup_weakrefs(val)
                if weak_id is not None:
                    self.id_matched_objs[local_name] = weak_id

    def NAME_MATCH(self, guard: Guard):
        if False:
            for i in range(10):
                print('nop')
        obj = self.get(guard.name)
        code = f"{self.arg_ref(guard)}.__name__ == '{obj.__name__}'"
        self._produce_guard_code(guard, [code])

    def DATA_PTR_MATCH(self, guard: Guard):
        if False:
            return 10
        obj = self.get(guard.name)
        code = f'{self.arg_ref(guard)}.data_ptr() == {obj.data_ptr()}'
        self._produce_guard_code(guard, [code])

    def HASATTR(self, guard: Guard):
        if False:
            for i in range(10):
                print('nop')
        m = re.match('^(.*)[.]([a-zA-Z0-9_]+)$', guard.name)
        assert m, f'invalid hasattr check {guard.name}'
        (base, attr) = m.group(1, 2)
        ref = self.arg_ref(base)
        val = hasattr(self.get(base), attr)
        code = None
        if val:
            code = f'hasattr({ref}, {attr!r})'
        else:
            code = f'not hasattr({ref}, {attr!r})'
        self._produce_guard_code(guard, [code], provided_guarded_object=self.get(base))

    def EQUALS_MATCH(self, guard: Guard):
        if False:
            for i in range(10):
                print('nop')
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        t = type(val)
        if np:
            np_types = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)
        else:
            np_types = ()
        ok_types = (int, float, bool, type(None), str, type, list, tuple, set, slice, frozenset, range, torch.Size, torch.device, torch.dtype, *np_types)
        if istype(val, dict):
            assert all((istype(x, ok_types) for x in itertools.chain(val.keys(), val.values())))
        else:
            assert istype(val, ok_types), t.__name__
        if istype(val, float) and math.isnan(val):
            code = list()
            code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
            code.append(f'__math_isnan({ref})')
            self._produce_guard_code(guard, code)
            return
        code = list()
        if istype(val, (list, tuple)):
            self.LIST_LENGTH(guard)
            for (idx, elem) in enumerate(val):
                code.append(f'___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})')
        else:
            code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        if istype(val, torch.Size):
            val = tuple(val)
        code.append(f'{ref} == {val!r}')
        self._produce_guard_code(guard, code)

    def CONSTANT_MATCH(self, guard: Guard):
        if False:
            i = 10
            return i + 15
        val = self.get(guard.name)
        if istype(val, (bool, type(None))):
            self.ID_MATCH(guard)
        else:
            self.EQUALS_MATCH(guard)

    def NN_MODULE(self, guard: Guard):
        if False:
            for i in range(10):
                print('nop')
        self.ID_MATCH(guard)
        ref = self.arg_ref(guard)
        val = self.get(guard.name)

        def setup_guard():
            if False:
                return 10
            assert istype(val.training, bool)
            self.code.append(GuardCodeList([f'{ref}.training == {val.training}'], guard))
        if hasattr(val, 'training'):
            setup_guard()
        else:
            exc.unimplemented(f'Guard setup for uninitialized class {type(val)}')

    def FUNCTION_MATCH(self, guard: Guard):
        if False:
            print('Hello World!')
        'things like torch.add and user defined functions'
        if guard.is_local():
            return self.ID_MATCH(guard)

    def CLOSURE_MATCH(self, guard: Guard):
        if False:
            while True:
                i = 10
        'matches a closure by __code__ id.'
        if guard.is_local():
            val = self.get(guard.name)
            if type(val) == types.FunctionType and hasattr(val, '__code__'):
                ref = self.arg_ref(guard)
                code = [f"___check_obj_id(getattr({ref}, '__code__', None), {self.id_ref(val.__code__)})"]
                self._produce_guard_code(guard, code)
            else:
                self.FUNCTION_MATCH(guard)

    def BUILTIN_MATCH(self, guard: Guard):
        if False:
            print('Hello World!')
        return self.FUNCTION_MATCH(guard)

    def PYMODULE_MATCH(self, guard: Guard):
        if False:
            i = 10
            return i + 15
        return self.FUNCTION_MATCH(guard)

    def LIST_LENGTH(self, guard):
        if False:
            return 10
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'len({ref}) == {len(value)}')
        self._produce_guard_code(guard, code)

    def TUPLE_ITERATOR_LEN(self, guard):
        if False:
            return 10
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'___tuple_iterator_len({ref}) == {tuple_iterator_len(value)}')
        self._produce_guard_code(guard, code)

    def DUPLICATE_INPUT(self, guard, source_b):
        if False:
            i = 10
            return i + 15
        ref_a = self.arg_ref(guard)
        ref_b = self.arg_ref(source_b.name())
        code = [f'{ref_b} is {ref_a}']
        self._produce_guard_code(guard, code)

    def DICT_KEYS(self, guard):
        if False:
            print('Hello World!')
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        param_key_ids = set(dict_param_key_ids(value))
        const_keys = set(dict_const_keys(value))
        const_keys_repr = dict_const_keys_repr(const_keys, local=is_from_local_source(guard.originating_source))
        if param_key_ids:
            code.append(f'___dict_param_key_ids({ref}) == {param_key_ids!r}')
            code.append(f'___dict_const_keys({ref}) == {const_keys_repr}')
        else:
            code.append(f'set({ref}.keys()) == {const_keys_repr}')
        self._produce_guard_code(guard, code)

    def WEAKREF_ALIVE(self, guard):
        if False:
            return 10
        self._produce_guard_code(guard, [f'{self.arg_ref(guard)} is not None'])

    def NN_MODULE_PARAM_NAMES(self, guard):
        if False:
            while True:
                i = 10
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        keys = {k for (k, v) in value.named_parameters()}
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'{{k for k, v in {ref}.named_parameters()}} == {keys!r}')
        self._produce_guard_code(guard, code)

    def ODICT_KEYS(self, guard):
        if False:
            for i in range(10):
                print('nop')
        'OrderedDict keys match'
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'str({ref}.keys()) == {str(value.keys())!r}')
        self._produce_guard_code(guard, code)

    def OBJECT_MUTATION(self, guard: Guard):
        if False:
            return 10
        mutation_guard.watch(self.get(guard.name), self.check_fn_manager)

    def GRAD_MODE(self, guard: Guard):
        if False:
            i = 10
            return i + 15
        pass

    def DETERMINISTIC_ALGORITHMS(self, guard: Guard):
        if False:
            return 10
        pass

    def TORCH_FUNCTION_STATE(self, guard: Guard):
        if False:
            i = 10
            return i + 15
        pass

    def DEFAULT_DEVICE(self, guard: Guard):
        if False:
            while True:
                i = 10
        'Guard on CURRENT_DEVICE per torch.utils._device'
        assert guard.source is GuardSource.GLOBAL
        import torch.utils._device as m
        self._produce_guard_code(guard, [f'utils_device.CURRENT_DEVICE == {m.CURRENT_DEVICE!r}'])

    def BACKEND_MATCH(self, guard: Guard):
        if False:
            return 10
        'Guard on backend matching based on id of current_backend'
        assert guard.source is GuardSource.GLOBAL
        backend_id = f'{id(torch._dynamo.eval_frame.guarded_backend_cache.current_backend)}'
        code = [f'(___skip_backend_check() or ___current_backend() == ___lookup_backend({backend_id}))']
        self._produce_guard_code(guard, code)

    def SHAPE_ENV(self, guard: Guard):
        if False:
            i = 10
            return i + 15
        assert guard.name == ''
        output_graph = self.check_fn_manager.output_graph
        fs = output_graph.tracked_fakes
        constraint_inputs = [a.constraint_dims for a in fs]

        def get_sources(t_id, dim):
            if False:
                while True:
                    i = 10
            return [TensorPropertySource(source, TensorProperty.SIZE, dim) for source in output_graph.tracked_fakes_id_to_source[t_id]]
        if output_graph.export_constraints:
            source_pairs: List[Tuple[Source, Source]] = []
            for constraint in output_graph.export_constraints:
                if constraint.t_id in output_graph.tracked_fakes_id_to_source:
                    (source, *other_sources) = get_sources(constraint.t_id, constraint.dim)
                    source_pairs.extend(((source, other_source) for other_source in other_sources))
                    if constraint.shared is not None:
                        other_sources = get_sources(constraint.shared.t_id, constraint.shared.dim)
                        source_pairs.extend(((source, other_source) for other_source in other_sources))
                else:
                    log.warning('Untracked tensor used in export constraints')
            equalities_inputs = EqualityConstraint(source_pairs=source_pairs, warn_only=False)
        else:
            equalities_inputs = None
        guards = output_graph.shape_env.produce_guards([a.fake for a in fs], [a.source for a in fs], constraint_inputs=constraint_inputs, equalities_inputs=equalities_inputs, source_ref=self.source_ref, ignore_static=not self.check_fn_manager.output_graph.export)
        output_graph.shape_env.freeze()
        for shape_guard in guards:
            self._produce_guard_code(guard, [shape_guard], shape_env=True)

    def TENSOR_MATCH(self, guard: Guard, value=None):
        if False:
            return 10
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            if isinstance(value, TensorWeakRef):
                value = value()
            value = value if value is not None else self.get(guard.name)
            assert isinstance(value, torch.Tensor)
            tensor_name = self.arg_ref(guard)
            code: List[str] = list()
            if self.check_fn_manager.output_graph.export:
                self.TYPE_MATCH(guard)
                terms = ['dtype', 'device', 'requires_grad', 'ndimension()']
                for term in terms:
                    real_value = self.get(tensor_name + '.' + term)
                    if istype(real_value, (torch.device, torch.dtype)):
                        code.append(f'str({tensor_name}.{term}) == {str(real_value)!r}')
                    else:
                        code.append(f'{tensor_name}.{term} == {real_value}')
            else:
                self.tensor_check_names.append(tensor_name)
                self.tensor_check_examples.append(value)
                self.tensor_check_guards.append(guard)
            assert guard.source is not None
            (static, reason) = tensor_always_has_static_shape(value, is_tensor=True, guard_source=guard.source)
            if not static:
                if hasattr(value, '_dynamo_dynamic_indices'):
                    code.append(f"(({tensor_name}._dynamo_dynamic_indices.issubset({value._dynamo_dynamic_indices})) if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)")
                else:
                    code.append(f"hasattr({tensor_name}, '_dynamo_dynamic_indices') == False")
            if len(code) > 0:
                self._produce_guard_code(guard, code)

    def _produce_guard_code(self, guard, code_list, provided_guarded_object=None, shape_env=False):
        if False:
            for i in range(10):
                print('nop')
        cur_frame = currentframe()
        assert cur_frame is not None
        caller = cur_frame.f_back
        del cur_frame
        assert caller is not None
        func_name = getframeinfo(caller)[2]
        del caller
        assert func_name in dir(self.__class__), f'_produce_guard_code must be called from inside GuardedCode. Called from {func_name}'
        if shape_env:
            self.shape_env_code.append(GuardCodeList(code_list, guard))
        else:
            self.code.append(GuardCodeList(code_list, guard))
        if provided_guarded_object is None:
            name_valid = guard.name is not None and guard.name != ''
            guarded_object = self.get(guard.name) if name_valid else None
        else:
            guarded_object = provided_guarded_object
        guarded_object_type = weakref.ref(type(guarded_object)) if guarded_object is not None else None
        obj_ref = None
        if hasattr(guarded_object.__class__, '__weakref__') and (not isinstance(guarded_object, enum.Enum)):
            obj_ref = weakref.ref(guarded_object)
        guard.set_export_info(func_name, guarded_object_type, code_list, obj_ref)

class PyExprCSEPass:
    USE_THRESHOLD = 1
    ALLOWED_NODE_TYPES = (ast.Attribute, ast.Call, ast.Subscript)

    @dataclasses.dataclass
    class Config:
        expr_count: Dict[str, int]
        expr_to_name: Dict[str, str]

    class ExprCounter(ast.NodeVisitor):

        def __init__(self, config: PyExprCSEPass.Config) -> None:
            if False:
                while True:
                    i = 10
            self._config = config

        def visit(self, node: ast.AST) -> Any:
            if False:
                i = 10
                return i + 15
            if isinstance(node, PyExprCSEPass.ALLOWED_NODE_TYPES):
                self._config.expr_count[_ast_unparse(node)] += 1
            super().visit(node)

    class Replacer(ast.NodeTransformer):

        def __init__(self, config: PyExprCSEPass.Config, gen_name: Callable[[], str]) -> None:
            if False:
                while True:
                    i = 10
            super().__init__()
            self._config = config
            self._gen_name = gen_name
            self.preface: List[str] = []

        def visit(self, node: ast.AST) -> Any:
            if False:
                while True:
                    i = 10
            if isinstance(node, PyExprCSEPass.ALLOWED_NODE_TYPES):
                expr = _ast_unparse(node)
                if self._config.expr_count[expr] > PyExprCSEPass.USE_THRESHOLD:
                    if expr not in self._config.expr_to_name:
                        node_ = super().visit(node)
                        expr_ = _ast_unparse(node_)
                        var_name = self._gen_name()
                        self.preface.append(f'{var_name} = {expr_}')
                        self._config.expr_to_name[expr] = var_name
                    else:
                        var_name = self._config.expr_to_name[expr]
                    return ast.Name(var_name, ast.Load())
            return super().visit(node)

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._counter = 0
        self._config = self.Config(expr_count=collections.defaultdict(lambda : 0), expr_to_name={})

    def _new_var(self, prefix: str='_var') -> str:
        if False:
            return 10
        name = f'{prefix}{self._counter}'
        self._counter += 1
        return name

    def count(self, exprs: List[str]) -> None:
        if False:
            print('Hello World!')
        counter = self.ExprCounter(self._config)
        for e in exprs:
            counter.visit(ast.parse(e))

    def replace(self, expr: str) -> Tuple[List[str], str]:
        if False:
            i = 10
            return i + 15
        replacer = self.Replacer(self._config, self._new_var)
        new_node = replacer.visit(ast.parse(expr))
        return (replacer.preface, _ast_unparse(new_node))

def must_add_nn_module_guards(guard):
    if False:
        print('Hello World!')
    return isinstance(guard.originating_source, DefaultsSource) or (config.guard_nn_modules_using_dict_tags and guard.create_fn is GuardBuilder.NN_MODULE)

class CheckFunctionManager:

    def __init__(self, output_graph=None, guard_fail_fn: Optional[Callable[[GuardFail], None]]=None):
        if False:
            print('Hello World!')
        guards = output_graph.guards if output_graph else None
        self.valid = True
        self._weakrefs: Dict[int, ReferenceType[object]] = {}
        self.output_graph = output_graph

        def combine_scopes(left, right):
            if False:
                print('Hello World!')
            if left is None:
                return right
            if right is None:
                return left
            return {**left, **right}
        w_builder = None

        def source_ref(source):
            if False:
                for i in range(10):
                    print('nop')
            guard_source = source.guard_source()
            if guard_source is GuardSource.CONSTANT:
                return source.name()
            assert w_builder
            r_builder = w_builder()
            assert r_builder is not None
            return r_builder.arg_ref(source.name())
        builder = GuardBuilder(self.id_ref, source_ref, self.lookup_weakrefs, output_graph.local_scope, output_graph.global_scope, self)
        w_builder = weakref.ref(builder)
        for guard in sorted(guards or [], key=Guard.sort_key):
            if not config.guard_nn_modules and guard.is_nn_module() and ('__defaults__' not in guard.name) and ('__kwdefaults__' not in guard.name) and (config.skip_nnmodule_hook_guards or 'hooks' not in guard.name):
                continue
            guard.create(builder)
        self.check_fn = self.compile_check_fn(builder, guards, guard_fail_fn)
        self._weakrefs.clear()
        self.check_fn.id_matched_objs = builder.id_matched_objs

    def compile_check_fn(self, builder, guards_out, guard_fail_fn):
        if False:
            i = 10
            return i + 15
        largs = builder.argnames
        largs += ['**___kwargs_ignored']
        guards_log.debug('GUARDS:')
        code_parts = ['___guarded_code.valid', '___check_global_state()']

        def add_code_part(code, guard, log_only=False):
            if False:
                while True:
                    i = 10
            extra = ''
            if guard.user_stack:
                for fs in reversed(guard.user_stack):
                    if fs.filename not in uninteresting_files():
                        break
                else:
                    extra = f'  # {format_frame(fs, line=True)}'
            elif guard.stack:
                extra = f'  # {format_frame(guard.stack.summary()[-1])}'
            guards_log.debug('%s', f'{code:<60}{extra}')
            if verbose_guards_log.isEnabledFor(logging.DEBUG):
                maybe_stack = ''
                maybe_user_stack = ''
                if guard is not None:
                    if guard.stack:
                        maybe_stack = f"\nStack:\n{''.join(guard.stack.format())}"
                    if guard.user_stack:
                        maybe_user_stack = f"\nUser stack:\n{''.join(guard.user_stack.format())}"
                verbose_guards_log.debug('Guard: %s%s%s', code, maybe_stack, maybe_user_stack)
            if not log_only:
                code_parts.append(code)
        seen = set()
        for gcl in builder.code:
            for code in gcl.code_list:
                if code not in seen:
                    add_code_part(code, gcl.guard)
                    seen.add(code)
        tensor_check_names = builder.tensor_check_names
        check_tensors_fn = None
        check_tensors_verbose_fn = None
        if tensor_check_names:
            assert not self.output_graph.export, 'Illegal to set tensor_check_names in export.'
            tensor_check_examples = builder.tensor_check_examples

            def convert(size_or_stride):
                if False:
                    while True:
                        i = 10
                converted: List[Optional[int]] = []
                for dim in size_or_stride:
                    if not is_symbolic(dim):
                        converted.append(dim)
                    else:
                        assert isinstance(dim, torch.SymInt)
                        converted.append(dim.node.maybe_as_int())
                return converted
            dynamic_dims_sizes = [convert(self.output_graph.tensor_weakref_to_sizes_strides[t]['size']) for t in tensor_check_examples]
            dynamic_dims_strides = [convert(self.output_graph.tensor_weakref_to_sizes_strides[t]['stride']) for t in tensor_check_examples]
            tensor_guards = TensorGuards(*tensor_check_examples, dynamic_dims_sizes=dynamic_dims_sizes, dynamic_dims_strides=dynamic_dims_strides)
            check_tensors_fn = tensor_guards.check
            check_tensors_verbose_fn = tensor_guards.check_verbose
            tensor_check_args = ', '.join(tensor_check_names + ['tensor_check_names=tensor_check_names'])
            code_parts.append(f'___check_tensors({tensor_check_args})')
            tensor_check_guards = builder.tensor_check_guards
            for (i, name) in enumerate(tensor_check_names):
                t = tensor_check_examples[i]
                pytype = type(t)
                dispatch_key = (torch._C._dispatch_keys(t) | torch._C._dispatch_tls_local_include_set()) - torch._C._dispatch_tls_local_exclude_set()
                dtype = t.dtype
                device_index = t.device.index
                requires_grad = t.requires_grad
                sizes = dynamic_dims_sizes[i]
                strides = dynamic_dims_strides[i]
                add_code_part(f'check_tensor({name}, {pytype.__qualname__}, {dispatch_key}, {dtype}, device={device_index}, requires_grad={requires_grad}, size={sizes}, stride={strides})', tensor_check_guards[i], log_only=True)
        aotautograd_guards: List[GuardEnvExpr] = self.output_graph.tracing_context.guards_context.aotautograd_guards if self.output_graph else []
        for guard in aotautograd_guards:
            if isinstance(guard, DuplicateInputs):
                source_a = guard.input_source_a
                source_b = guard.input_source_b
                add_code_part(f'{source_a.name()} is {source_b.name()}', None)
            else:
                raise RuntimeError(f'Unknown GuardEnvExpr: {guard}')
        for gcl in builder.shape_env_code:
            for code in gcl.code_list:
                add_code_part(code, gcl.guard)
        global_state = convert_frame.initial_global_state
        if global_state is None:
            global_state = convert_frame.GlobalStateGuard()
        closure_vars = {'___guarded_code': self, '___check_tensors': check_tensors_fn, '___check_tensors_verbose': check_tensors_verbose_fn, '___check_global_state': global_state.check, 'tensor_check_names': tensor_check_names, **SYMPY_INTERP, **CLOSURE_VARS}
        unique_code_parts = list(unique(code_parts))
        make_guard_fn_args = ', '.join(closure_vars.keys())
        (guard_body, pycode) = build_guard_function(unique_code_parts, make_guard_fn_args)
        if os.environ.get('TORCHDYNAMO_PRINT_GUARDS', None) == '1':
            print('GUARDS\n', guard_body)
        out: Dict[str, Any] = dict()
        exec(pycode, builder.scope, out)
        guard_fn = out['___make_guard_fn'](*closure_vars.values())
        guard_fn.closure_vars = closure_vars
        guard_fn.args = largs
        guard_fn.code_parts = code_parts
        guard_fn.global_scope = {'G': builder.scope['G']}
        guard_fn.guard_fail_fn = guard_fail_fn
        return guard_fn

    def invalidate(self):
        if False:
            print('Hello World!')
        self.valid = False

    def id_ref(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'add a weakref, return the id'
        try:
            if id(obj) not in self._weakrefs:
                self._weakrefs[id(obj)] = weakref.ref(obj)
                weakref.finalize(obj, self.invalidate)
        except TypeError:
            pass
        return id(obj)

    def lookup_weakrefs(self, obj):
        if False:
            print('Hello World!')
        "Lookup the _weakrefs created in id_ref function for ID_MATCH'd objects"
        if id(obj) in self._weakrefs:
            return self._weakrefs[id(obj)]
        return None

def build_guard_function(code_parts, closure_args) -> Tuple[str, str]:
    if False:
        i = 10
        return i + 15
    from torch._inductor.utils import IndentedBuffer
    if HAS_UNPARSE_FUNCTIONS:
        csepass = PyExprCSEPass()
        csepass.count(code_parts)

        def replace(expr: str) -> Tuple[List[str], str]:
            if False:
                print('Hello World!')
            return csepass.replace(expr)
    else:

        def replace(expr: str) -> Tuple[List[str], str]:
            if False:
                print('Hello World!')
            return ([], expr)
    guard_body = IndentedBuffer()
    for expr in code_parts:
        (preface, expr) = replace(expr)
        guard_body.writelines(preface)
        guard_body.writeline(f'if not ({expr}):')
        with guard_body.indent():
            guard_body.writeline('return False')
    guard = IndentedBuffer()
    guard.writeline('def guard(L):')
    with guard.indent():
        guard.splice(guard_body)
        guard.writeline('return True')
    make_guard_fn = IndentedBuffer()
    make_guard_fn.writeline(f'def ___make_guard_fn({closure_args}):')
    with make_guard_fn.indent():
        make_guard_fn.splice(guard)
        make_guard_fn.writeline('return guard')
    return (guard_body.getvalue(), make_guard_fn.getvalue())

def get_guard_fail_reason(guard_fn: GuardFn, code: types.CodeType, f_locals: Dict[str, object]) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the reason why `guard_fn` failed.\n    Updates `guard_failures` with the generated reason.\n    Only the first failed check of guard_fn is reported.\n    '
    scope = {'L': f_locals, 'G': guard_fn.global_scope['G']}
    scope.update(guard_fn.closure_vars)
    scope['___check_tensors'] = scope['___check_tensors_verbose']
    reason = ''
    for part in guard_fn.code_parts:
        global_scope = dict(guard_fn.global_scope)
        global_scope['__compile_source__'] = part
        with report_compile_source_on_error():
            fail_reason = eval(part, global_scope, scope)
        if isinstance(fail_reason, bool) and (not fail_reason):
            reason = part
            break
        elif isinstance(fail_reason, str):
            reason = fail_reason
            break
    guard_failures[orig_code_map[code]].append(reason)
    try:
        if guard_fn.guard_fail_fn is not None:
            guard_fn.guard_fail_fn(GuardFail(reason or 'unknown reason', orig_code_map[code]))
    except Exception as e:
        log.error('Failure in guard_fail_fn callback - raising here will cause a NULL Error on guard eval', exc_info=True)
    return reason

def get_and_maybe_log_recompilation_reason(cache_entry, frame: types.FrameType) -> List[str]:
    if False:
        return 10
    '\n    Return the list of guard failure reasons using cache_entry.\n    Logs the recompilation reason if `recompiles` logging is enabled.\n    Raises a RecompileError if `config.error_on_recompile` is enabled.\n    '
    reasons = []
    while cache_entry is not None:
        reason = get_guard_fail_reason(cache_entry.check_fn, cache_entry.code, frame.f_locals)
        if reason:
            reasons.append(reason)
        cache_entry = cache_entry.next
    code = frame.f_code
    do_recompiles_log = is_guard_failure_reporting_enabled() and recompiles_log.isEnabledFor(logging.DEBUG)
    if do_recompiles_log or config.error_on_recompile:
        failures = '\n'.join(reasons)
        guard_failure_details = f"triggered by the following guard failure(s):\n{textwrap.indent(failures, '- ')}"
        message = f"Recompiling function {code.co_name} in {code.co_filename}:{code.co_firstlineno}\n{textwrap.indent(guard_failure_details, '    ')}"
        if do_recompiles_log:
            recompiles_log.debug(message, stack_info=True)
        if config.error_on_recompile:
            raise exc.RecompileError(message)
    return reasons

def guard_error_hook(guard_fn: GuardFn, code: types.CodeType, f_locals: Dict[str, object], index: int, last: bool):
    if False:
        print('Hello World!')
    print(f'ERROR RUNNING GUARDS {code.co_name} {code.co_filename}:{code.co_firstlineno}')
    print('lambda ' + ', '.join(guard_fn.args) + ':')
    print(' ', ' and\n  '.join(guard_fn.code_parts))
set_guard_error_hook(guard_error_hook)

def unique(seq):
    if False:
        print('Hello World!')
    seen = set()
    for x in seq:
        if x not in seen:
            yield x
            seen.add(x)

def make_dupe_guard(obj_source, dupe_source):
    if False:
        print('Hello World!')
    if dupe_source and dupe_source != obj_source:
        ser_source_is_local = is_from_local_source(dupe_source)
        source_is_local = is_from_local_source(obj_source)
        if ser_source_is_local == source_is_local:
            return functools.partial(GuardBuilder.DUPLICATE_INPUT, source_b=dupe_source)
    return None

def install_guard(*guards, skip=0):
    if False:
        return 10
    '\n    Add dynamo guards to the current tracing context.\n\n    Args:\n        guards: guard(s) to add\n        skip: number of stack frames to ignore for debug stack trace\n    '
    from torch._guards import TracingContext
    add = TracingContext.get().guards_context.dynamo_guards.add
    for guard in guards:
        assert isinstance(guard, Guard)
        add(guard, skip=skip + 1)