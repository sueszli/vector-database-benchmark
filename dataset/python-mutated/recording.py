import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
__all__ = ['ShapeEnvEvent', 'record_shapeenv_event', 'replay_shape_env_events', 'FakeTensorMeta', 'shape_env_check_state_equal', 'NotEqualError']

@dataclass
class ShapeEnvEvent:
    f: Callable
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    tracked_fakes: Optional[List[Any]] = None
    name: Optional[str] = None

    def run(self, shape_env=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymTypes
        if self.f is ShapeEnv:
            assert shape_env is None and self.args is None and (self.kwargs is not None)
            return ShapeEnv(**self.kwargs)
        assert shape_env is not None
        args = list(self.args or list())
        kwargs = dict(self.kwargs or dict())
        (args, kwargs) = pytree.tree_map_only(ShapeEnv, lambda _: shape_env, (args, kwargs))
        (args, kwargs) = pytree.tree_map_only(SymTypes, lambda a: type(a)(a.node.with_shape_env(shape_env)), (args, kwargs))

        def maybe_convert_node(x: Any) -> Any:
            if False:
                i = 10
                return i + 15
            if not isinstance(x, torch.fx.Node):
                return x
            assert hasattr(shape_env, 'name_to_node')
            name_to_node = shape_env.name_to_node
            assert x.name in name_to_node
            return name_to_node[x.name]

        def replacearg(index: int, key: str, fn: Callable):
            if False:
                print('Hello World!')
            if index < len(args):
                args[index] = fn(args[index])
            if key in kwargs:
                kwargs[key] = fn(kwargs[key])
        if self.is_create_fx_call_function():
            replacearg(index=2, key='args', fn=lambda args: tuple((maybe_convert_node(a) for a in args)))
        if self.is_evaluate_expr() or self.is_defer_runtime_assert():
            replacearg(index=3, key='fx_node', fn=maybe_convert_node)
        return self.f(*args, **kwargs)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        name = self.name if self.name is not None else self.f.__name__
        return f'event: {name} ({self.args}, {self.kwargs})'

    def is_create_fx_call_function(self) -> bool:
        if False:
            return 10
        return self.name == 'create_fx_call_function'

    def is_evaluate_expr(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.name == 'evaluate_expr'

    def is_defer_runtime_assert(self) -> bool:
        if False:
            while True:
                i = 10
        return self.name == 'defer_runtime_assert'

def _extract_shape_env_and_assert_equal(args, kwargs):
    if False:
        print('Hello World!')
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymTypes

    def assert_equal(old: Optional[ShapeEnv], new: ShapeEnv) -> ShapeEnv:
        if False:
            return 10
        if old is not None:
            assert old is new, 'call with different ShapeEnv'
        return new
    shape_env = None
    for val in itertools.chain(args, kwargs.values()):
        if isinstance(val, ShapeEnv):
            shape_env = assert_equal(shape_env, val)
        if isinstance(val, SymTypes):
            shape_env = assert_equal(shape_env, val.node.shape_env)
    return shape_env

def record_shapeenv_event(*, save_tracked_fakes: bool=False) -> Callable:
    if False:
        print('Hello World!')

    def decorator(fn: Callable) -> Callable:
        if False:
            i = 10
            return i + 15
        assert callable(fn)
        name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            if isinstance(args[0], ShapeEnv) and args[0].is_recording:
                return fn(*args, **kwargs)
            self = _extract_shape_env_and_assert_equal(args, kwargs)
            if self is None:
                return fn(*args, **kwargs)
            with self.recording():
                tracked_fakes = self.snapshot_tracked_fakes() if save_tracked_fakes else None
                event = ShapeEnvEvent(fn, list(args), kwargs, tracked_fakes, name=fn.__name__)
                self.events.append(event)
                return event.run(self)
        return wrapper
    return decorator

def replay_shape_env_events(events):
    if False:
        for i in range(10):
            print('nop')
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    constructor_event = events[0]
    assert constructor_event.f == ShapeEnv
    shape_env = constructor_event.run()
    for event in events[1:]:
        try:
            event.run(shape_env)
        except Exception as e:
            raise RuntimeError(f'failed when running event: {event}') from e
    return shape_env

@dataclass
class FakeTensorMeta:
    tensor_size: Tuple[Union[int, torch.SymInt], ...]
    tensor_stride: Tuple[Union[int, torch.SymInt], ...]
    tensor_storage_offset: Union[int, torch.SymInt]
    is_nested: bool

    def size(self) -> Tuple[Union[int, torch.SymInt], ...]:
        if False:
            for i in range(10):
                print('nop')
        return self.tensor_size

    def stride(self) -> Tuple[Union[int, torch.SymInt], ...]:
        if False:
            for i in range(10):
                print('nop')
        return self.tensor_stride

    def storage_offset(self) -> Union[int, torch.SymInt]:
        if False:
            for i in range(10):
                print('nop')
        return self.tensor_storage_offset

    def dim(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.tensor_size)

    @staticmethod
    def from_fake(fake) -> 'FakeTensorMeta':
        if False:
            i = 10
            return i + 15
        return FakeTensorMeta(fake.size(), fake.stride(), fake.storage_offset(), fake.is_nested)

def shape_env_check_state_equal(env1, env2, non_state_variable_names, map_value):
    if False:
        return 10
    env1_vars = vars(env1).copy()
    env2_vars = vars(env2).copy()
    for v in non_state_variable_names:
        if v in env1_vars:
            env1_vars.pop(v)
        if v in env2_vars:
            env2_vars.pop(v)

    def value_to_str(value: Any) -> str:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, dict):
            return '{' + ', '.join((f'{k}: {value[k]}' for k in sorted(value.keys(), key=str))) + '}'
        if isinstance(value, set):
            return '{' + ', '.join((f'{v}' for v in sorted(value))) + '}'
        return str(value)

    def compare_vars(map_value: Callable[[str, Any], Any]) -> List[Tuple[str, str, str]]:
        if False:
            return 10
        (env1_set, env2_set) = (set(env1_vars), set(env2_vars))
        if env1_set != env2_set:
            raise NotEqualError('field set mismatch:', [('found unique fields:', str(sorted(env1_set - env2_set)), str(sorted(env2_set - env1_set)))])
        sorted_keys = list(env1_set)
        sorted_keys.sort()
        mapped_dict = [(k, map_value(k, env1_vars[k]), map_value(k, env2_vars[k])) for k in sorted_keys]
        return [(f"{k}: values don't match.", value_to_str(val1), value_to_str(val2)) for (k, val1, val2) in mapped_dict if val1 != val2]
    errors = compare_vars(map_value)
    if len(errors) > 0:
        raise NotEqualError("field values don't match:", errors)

class NotEqualError(Exception):

    def __init__(self, msg: str, mismatched: List[Tuple[str, str, str]]) -> None:
        if False:
            while True:
                i = 10
        details = '\n'.join(['\n'.join([f'==> {inner_msg}', f'  >  Left: {str1}', f'  > Right: {str2}']) for (inner_msg, str1, str2) in mismatched])
        super().__init__(f'ShapeEnv not equal: {msg}\n\n{details}\n')