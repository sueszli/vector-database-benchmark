from __future__ import annotations
import contextlib
import dataclasses
import enum
import functools
import logging
import threading
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generic, List, NamedTuple, Optional, Set, Tuple, TYPE_CHECKING, TypeVar
import torch
from torch.utils import _pytree as pytree
from torch.utils._traceback import CapturedTraceback
log = logging.getLogger(__name__)
if TYPE_CHECKING:
    import sympy
'\ntorch._guards is the definitional source of truth for general purpose guard structures.\n\nAn important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,\nand no guard installation notions here.\n'

class CompileId(NamedTuple):
    frame_id: int
    frame_compile_id: int

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'{self.frame_id}/{self.frame_compile_id}'

class TraceId(NamedTuple):
    compile_id: CompileId
    attempt: int

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f'{self.compile_id}_{self.attempt}'

class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8

    def is_fsdp_module(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self in (GuardSource.GLOBAL_FSDP_MODULE, GuardSource.LOCAL_FSDP_MODULE)

    def is_nn_module(self) -> bool:
        if False:
            while True:
                i = 10
        return self in (GuardSource.GLOBAL_NN_MODULE, GuardSource.LOCAL_NN_MODULE) or self.is_fsdp_module()

    def is_local(self):
        if False:
            i = 10
            return i + 15
        return self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE, GuardSource.LOCAL_FSDP_MODULE)
'\nBase class for a "GuardBuilder" role.\n\nThe GuardBuilderBase role is to represent a scope within which to build a guard. The name is a little\nconfusing, as its not a builder, but for the sake of avoiding a lot of renames and keeping the original reference\nto torchdynamo\'s GuardBuilder.\n\nNote: create_fn is invoked with a GuardBuilderBase and a Guard. A GuardBuilder is chosen based\non GuardSource\'s select function.\n\nThere is value in keeping this GuardBuilderBase empty to keep layering clean.\n'

class GuardBuilderBase:
    pass

class ShapeGuard(NamedTuple):
    expr: sympy.Expr
    stack: CapturedTraceback

@dataclasses.dataclass
class Guard:
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[type] = None
    stack = None
    user_stack = None
    _hash = None

    def __hash__(self):
        if False:
            print('Hello World!')
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self):
        if False:
            while True:
                i = 10
        return (self.source.value if self.source else -1, len(self.name), self.name, self.inner_create_fn().__code__.co_firstlineno)

    def __lt__(self, other):
        if False:
            return 10
        return self.sort_key() < other.sort_key()

    def inner_create_fn(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.create_fn, functools.partial):
            return self.create_fn.func
        else:
            return self.create_fn

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.originating_source.name()

    @property
    def source(self) -> GuardSource:
        if False:
            i = 10
            return i + 15
        return self.originating_source.guard_source()

    @staticmethod
    def weakref_to_str(obj_weakref):
        if False:
            while True:
                i = 10
        "\n        This is a workaround of a Python weakref bug.\n\n        `obj_weakref` is instance returned by `weakref.ref`,\n        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:\n\n            class MyConfig(dict):\n                def __getattr__(self, x):\n                    return self[x]\n\n            obj = MyConfig(offset=5)\n            obj_weakref = weakref.ref(obj)\n            str(obj_weakref)  # raise error: KeyError: '__name__'\n        "
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f'<weakref at {hex(id(obj_weakref))}; dead>'
        else:
            return str(obj_weakref)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        s = f"\n        {(self.source.name.lower() if self.source else '')} {repr(self.name)} {self.inner_create_fn().__name__}\n        {{\n            'guard_types': {self.guard_types},\n            'code': {self.code_list},\n            'obj_weakref': {self.weakref_to_str(self.obj_weakref)}\n            'guarded_class': {self.guarded_class_weakref}\n        }}\n        "
        return s

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        output = f'Name: {repr(self.name)}\n'
        source = self.source.name.lower() if self.source else ''
        output += f'    Source: {source}\n'
        output += f'    Create Function: {self.inner_create_fn().__name__}\n'
        output += f'    Guard Types: {self.guard_types}\n'
        output += f'    Code List: {self.code_list}\n'
        output += f'    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n'
        output += f'    Guarded Class Weakref: {self.guarded_class_weakref}\n'
        return output

    def create(self, builder: GuardBuilderBase):
        if False:
            print('Hello World!')
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.error('Error while creating guard:\n%s', str(self).rstrip())
            if self.stack:
                log.error('Created at:\n%s', ''.join(self.stack.format()[-4:]).rstrip())
            raise

    def is_nn_module(self):
        if False:
            print('Hello World!')
        return self.source.is_nn_module()

    def is_fsdp_module(self):
        if False:
            while True:
                i = 10
        return self.source.is_fsdp_module()

    def is_local(self):
        if False:
            i = 10
            return i + 15
        return self.source.is_local()

    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
        if False:
            return 10
        if not self.guard_types:
            self.guard_types = list()
        self.guard_types.append(guard_type)
        assert self.guarded_class_weakref in (guarded_class, None), 'Guarded class id must be identical, or None'
        self.guarded_class_weakref = guarded_class
        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)
        assert self.obj_weakref in (obj_weakref, None), 'Guarded object must be identical, or None'
        self.obj_weakref = obj_weakref
T = TypeVar('T')
'\nParent structure for guard env expressions.\nA GuardEnvExpr can have any subtype.\nNote: All subtypes must be handled exhaustively in\ntorch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.\n'

@dataclasses.dataclass
class GuardEnvExpr:
    pass
'\nA class representing a pair of duplicate inputs.\ninput_pos_a and input_pos_b are input positions we have deduped.\n'

@dataclasses.dataclass
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.input_source_a != self.input_source_b
'\nCheckpointable is an interface for driving state snapshotting, left purposely vague for now.\n\ncopy_graphstate() -> T, a somewhat legacy name, is expected to emit a snapshot of any type that\ncan also be taken in at restore_graphstate(T) calls.\n\nWhen to snapshot, is, at the moment, an implementation detail of upstream callers. Checkpointable\ndoes not provide any garuantees around consistency, idempotency, or safety of calling its APIs, yet.\n\nIn the future, it will have a closer coupling to a generic Checkpoint management system.\n'

class Checkpointable(ABC, Generic[T]):

    @abstractmethod
    def copy_graphstate(self) -> T:
        if False:
            i = 10
            return i + 15
        ...

    @abstractmethod
    def restore_graphstate(self, state: T):
        if False:
            print('Hello World!')
        ...
'\nThe GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext\n'

class GuardsCheckpointState:
    dynamo_guards: Set[Guard] = set()

    def __init__(self, dynamo_guards):
        if False:
            while True:
                i = 10
        self.dynamo_guards = dynamo_guards
    '\n    Produces a delta against another GuardsCheckpointState.\n\n    Returns None if no delta is found, otherwise, return a set() of mismatched\n    Guard type objects.\n    '

    def diff(self, other):
        if False:
            return 10
        r = self.dynamo_guards.difference(other.dynamo_guards)
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.diff(other) is None

class ModuleContextCheckpointState:
    nn_modules: Dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules):
        if False:
            for i in range(10):
                print('nop')
        self.nn_modules = nn_modules
    '\n    Produces a delta against another ModuleContextCheckpointState.\n\n    Returns None if no delta is found, otherwise, return a set() of mismatched\n    module key names.\n    '

    def diff(self, other):
        if False:
            print('Hello World!')
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.diff(other) is None

class ModuleContext(Checkpointable[ModuleContextCheckpointState]):

    def __init__(self):
        if False:
            print('Hello World!')
        self.nn_modules: Dict[str, Any] = {}

    def copy_graphstate(self):
        if False:
            i = 10
            return i + 15
        return ModuleContextCheckpointState(dict(self.nn_modules))

    def restore_graphstate(self, state):
        if False:
            return 10
        assert isinstance(state, ModuleContextCheckpointState)
        self.nn_modules = state.nn_modules

class GlobalContextCheckpointState:
    global_state: Dict[str, Tuple[Callable, ...]] = {}

    def __init__(self, global_states):
        if False:
            print('Hello World!')
        self.global_state = global_states
    '\n    Produces a delta against another GlobalContextCheckpointState.\n\n    Returns None if no delta is found, otherwise, return a set() of mismatched\n    global key names.\n    '

    def diff(self, other):
        if False:
            for i in range(10):
                print('nop')
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.diff(other) is None

class GlobalContext(Checkpointable[GlobalContextCheckpointState]):
    """
    This keeps track of the global torch state during tracing of a function.
    For example, torch.is_grad_enabled.
    """
    _supported_global_states = {'grad_enabled', 'torch_function_enabled', 'autocast_enabled', 'autocast_cpu_enabled', 'autocast_gpu_dtype', 'autocast_cpu_dtype', 'autocast_cache_enabled'}

    def __init__(self):
        if False:
            while True:
                i = 10
        self.global_state: Dict[str, Tuple[Callable, ...]] = {}

    def copy_graphstate(self):
        if False:
            return 10
        return GlobalContextCheckpointState(dict(self.global_state))

    def restore_graphstate(self, state):
        if False:
            i = 10
            return i + 15
        assert isinstance(state, GlobalContextCheckpointState)
        self.global_state = state.global_state
        assert len(self.global_state) == len(self._supported_global_states) and set(self.global_state.keys()) == self._supported_global_states, 'Global state mismatch'
        for (func, args) in self.global_state.values():
            func(args)
"\nA GuardsContext is a checkpointable representation of all the guards in the current tracing\ncontext. It's lifecycle is bound 1:1 to the tracing context, and it should never be instantiated\ndirectly outside of it. For passing around internal state representations of this object,\nprefer to extract them with copy_graphstate to produce a GuardsCheckpointState.\n"

class GuardsSet:

    def __init__(self, inner=None):
        if False:
            for i in range(10):
                print('nop')
        if inner is None:
            inner = set()
        self.inner = inner

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.inner)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.inner)

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return GuardsSet(self.inner - other.inner)

    def __bool__(self):
        if False:
            while True:
                i = 10
        return bool(self.inner)

    def add(self, guard: Guard, *, skip=0):
        if False:
            while True:
                i = 10
        if guard in self.inner:
            return
        if guard.stack is None:
            guard.stack = CapturedTraceback.extract(skip=1 + skip)
        if guard.user_stack is None:
            guard.user_stack = TracingContext.extract_stack()
        self.inner.add(guard)

    def update(self, *others: Set[Guard]):
        if False:
            for i in range(10):
                print('nop')
        for o in others:
            for g in o:
                self.add(g, skip=1)

class GuardsContext(Checkpointable[GuardsCheckpointState]):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.dynamo_guards: GuardsSet = GuardsSet()
        self.aotautograd_guards: List[GuardEnvExpr] = []

    def copy_graphstate(self):
        if False:
            return 10
        return GuardsCheckpointState(set(self.dynamo_guards.inner))

    def restore_graphstate(self, state):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(state, GuardsCheckpointState)
        self.dynamo_guards = GuardsSet(state.dynamo_guards)
_TLS = threading.local()
'\nTracingContext is the source of truth for all currently accumulated information\nneeded to trace. Its lifecycle is kept 1:1 when using TorchDynamo, but other systems\nare open to managing their own TracingContext with that in mind.\n\nThe purpose of TracingContext is not to be a dumping ground, or god object, but rather to avoid\nhaving to plumb complex subsystems across multiple verticals.\n\nEx: A common example is guard accumulation between dynamo, shape_env, aot_autograd, and inductor.\nAccessing the current tracing context via\nTracingContext.get() allows users to accumulate their own guards for processing, without needing to know how\nto plumb objects back up to where frame interpretation happened.\n\nNote that you can end up with multiple TracingContext for a single compilation\nof a frame, as we reset the TracingContext whenever we restart analysis.\nCompileContext is a more overarching context that encompasses multiple restarts.\n'

class CompileContext:

    @staticmethod
    def get() -> CompileContext:
        if False:
            for i in range(10):
                print('nop')
        assert _TLS.compile_context is not None
        return _TLS.compile_context

    @staticmethod
    def try_get() -> Optional[CompileContext]:
        if False:
            while True:
                i = 10
        return getattr(_TLS, 'compile_context', None)

    def __init__(self, compile_id):
        if False:
            return 10
        assert compile_id is None or isinstance(compile_id, CompileId)
        self.compile_id: Optional[CompileId] = compile_id
        self.attempt = 0

    @staticmethod
    def current_compile_id():
        if False:
            for i in range(10):
                print('nop')
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id():
        if False:
            i = 10
            return i + 15
        self = CompileContext.try_get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)

class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """

    @staticmethod
    def try_get() -> Optional[TracingContext]:
        if False:
            while True:
                i = 10
        return getattr(_TLS, 'tracing_context', None)

    @staticmethod
    def get() -> TracingContext:
        if False:
            for i in range(10):
                print('nop')
        if (ctx := TracingContext.try_get()):
            return ctx
        raise RuntimeError('TracingContext.get() must be called within an ongoing trace.')

    def __init__(self, fake_mode):
        if False:
            print('Hello World!')
        self.guards_context = GuardsContext()
        self.module_context = ModuleContext()
        self.global_context = GlobalContext()
        self.fake_mode = fake_mode
        self.frame_summary_stack = []
        self.loc_in_frame = None
        self.fw_metadata = None
        self.params_flat = None
        self.output_strides: Optional[List[Optional[List[int]]]] = None
        self.force_unspec_int_unbacked_size_like = False

    @staticmethod
    @contextmanager
    def patch(**kwargs):
        if False:
            print('Hello World!')
        prior = {}
        ctx = TracingContext.get()
        for key in kwargs.keys():
            prior[key] = getattr(ctx, key)
        for (key, val) in kwargs.items():
            setattr(ctx, key, val)
        try:
            yield
        finally:
            for (key, val) in prior.items():
                setattr(ctx, key, val)

    @staticmethod
    def extract_stack():
        if False:
            return 10
        self = TracingContext.try_get()
        if self is None:
            return traceback.StackSummary()
        stack = list(self.frame_summary_stack)
        if self.loc_in_frame is not None:
            stack.append(self.loc_in_frame)
        return traceback.StackSummary.from_list(stack)

    @staticmethod
    @contextlib.contextmanager
    def clear_frame():
        if False:
            while True:
                i = 10
        tc = TracingContext.get()
        with unittest.mock.patch.object(tc, 'frame_summary_stack', []), unittest.mock.patch.object(tc, 'loc_in_frame', None):
            try:
                yield
            except Exception as e:
                if not hasattr(e, 'real_stack'):
                    e.real_stack = None
                raise

    @staticmethod
    @contextlib.contextmanager
    def current_frame(frame_summary):
        if False:
            while True:
                i = 10
        tc = TracingContext.get()
        if frame_summary is not None:
            tc.frame_summary_stack.append(frame_summary)
        old = tc.loc_in_frame
        tc.loc_in_frame = None
        try:
            yield
        except Exception as e:
            if not hasattr(e, 'real_stack'):
                e.real_stack = tc.extract_stack()
            raise
        finally:
            if frame_summary is not None:
                tc.frame_summary_stack.pop()
            tc.loc_in_frame = old

    @staticmethod
    @contextlib.contextmanager
    def report_output_strides():
        if False:
            print('Hello World!')
        tc = TracingContext.try_get()
        if tc is None:
            yield None
            return
        old_output_strides = tc.output_strides
        tc.output_strides = []
        try:
            yield tc.output_strides
        finally:
            tc.output_strides = old_output_strides

    @staticmethod
    def set_current_loc(filename, lineno, frame_name):
        if False:
            return 10
        TracingContext.get().loc_in_frame = traceback.FrameSummary(filename, lineno, frame_name)

@contextmanager
def compile_context(context: CompileContext):
    if False:
        i = 10
        return i + 15
    old_context = getattr(_TLS, 'compile_context', None)
    _TLS.compile_context = context
    try:
        yield context
    finally:
        _TLS.compile_context = old_context

@contextmanager
def tracing(context: Optional[TracingContext]):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function installs the passed in tracing context as a dynamic scoped\n    global variable.\n\n    Calls to TracingContext.get() while not under a `with tracing()` context\n    will return None.\n    '
    old_context = getattr(_TLS, 'tracing_context', None)
    _TLS.tracing_context = context
    try:
        yield context
    except Exception as e:
        if not hasattr(e, 'real_stack') and context is not None:
            e.real_stack = context.extract_stack()
        raise
    finally:
        if context is not None and context.fake_mode is not None and (context.fake_mode.shape_env is not None):
            context.fake_mode.shape_env.cleanup()
        _TLS.tracing_context = old_context

@dataclasses.dataclass(frozen=True)
class Source:

    def reconstruct(self, codegen):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def guard_source(self) -> GuardSource:
        if False:
            return 10
        raise NotImplementedError()

    def name(self) -> str:
        if False:
            return 10
        raise NotImplementedError()

    def make_guard(self, fn) -> Guard:
        if False:
            while True:
                i = 10
        if self.guard_source() is GuardSource.CONSTANT:
            raise NotImplementedError()
        return Guard(self, fn)

    def is_nn_module(self) -> bool:
        if False:
            print('Hello World!')
        return self.guard_source().is_nn_module()

@dataclasses.dataclass(frozen=True)
class ChainedSource(Source):
    base: Source

def detect_fake_mode(inputs: Any=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attempts to "detect" what the current fake mode is.  If there is one ambiently\n    available from TracingContext, we preferentially use that.  Otherwise, we\n    heuristically detect the fake mode via the following sources, in order of\n    priority:\n\n        - Currently active fake mode on stack\n        - Fake mode associated with passed in tensors (inputs does not\n          have to be flattened)\n    '
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    fake_modes = []
    if (context := TracingContext.try_get()):
        fake_mode = context.fake_mode
        if fake_mode is not None:
            fake_modes.append((fake_mode, 'tracing context', 0))
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    for (i, m) in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            fake_modes.append((m, 'active fake mode', i))
    flat_inputs = pytree.tree_leaves(inputs)
    for (i, flat_input) in enumerate(flat_inputs):
        if isinstance(flat_input, FakeTensor):
            fake_modes.append((flat_input.fake_mode, 'fake tensor input', i))
    if fake_modes:
        (fake_mode, desc1, i1) = fake_modes[0]
        for (m, desc2, i2) in fake_modes[1:]:
            assert fake_mode is m, f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\nfake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\nfake mode from {desc2} {i2} allocated at:\n{m.stack}"
        return fake_mode
    else:
        return None