"""

The `Reactive` class implements [reactivity](/guide/reactivity/).
"""
from __future__ import annotations
from functools import partial
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Awaitable, Callable, ClassVar, Generic, Type, TypeVar
import rich.repr
from . import events
from ._callback import count_parameters
from ._types import MessageTarget, WatchCallbackType
if TYPE_CHECKING:
    from .dom import DOMNode
    Reactable = DOMNode
ReactiveType = TypeVar('ReactiveType')

class TooManyComputesError(Exception):
    """Raised when an attribute has public and private compute methods."""

@rich.repr.auto
class Reactive(Generic[ReactiveType]):
    """Reactive descriptor.

    Args:
        default: A default value or callable that returns a default.
        layout: Perform a layout on change.
        repaint: Perform a repaint on change.
        init: Call watchers on initialize (post mount).
        always_update: Call watchers even when the new value equals the old value.
        compute: Run compute methods when attribute is changed.
    """
    _reactives: ClassVar[dict[str, object]] = {}

    def __init__(self, default: ReactiveType | Callable[[], ReactiveType], *, layout: bool=False, repaint: bool=True, init: bool=False, always_update: bool=False, compute: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._default = default
        self._layout = layout
        self._repaint = repaint
        self._init = init
        self._always_update = always_update
        self._run_compute = compute

    def __rich_repr__(self) -> rich.repr.Result:
        if False:
            for i in range(10):
                print('nop')
        yield self._default
        yield ('layout', self._layout)
        yield ('repaint', self._repaint)
        yield ('init', self._init)
        yield ('always_update', self._always_update)
        yield ('compute', self._run_compute)

    def _initialize_reactive(self, obj: Reactable, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialized a reactive attribute on an object.\n\n        Args:\n            obj: An object with reactive attributes.\n            name: Name of attribute.\n        '
        _rich_traceback_omit = True
        internal_name = f'_reactive_{name}'
        if hasattr(obj, internal_name):
            return
        compute_method = getattr(obj, self.compute_name, None)
        if compute_method is not None and self._init:
            default = compute_method()
        else:
            default_or_callable = self._default
            default = default_or_callable() if callable(default_or_callable) else default_or_callable
        setattr(obj, internal_name, default)
        if self._init:
            self._check_watchers(obj, name, default)

    @classmethod
    def _initialize_object(cls, obj: Reactable) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set defaults and call any watchers / computes for the first time.\n\n        Args:\n            obj: An object with Reactive descriptors\n        '
        _rich_traceback_omit = True
        for (name, reactive) in obj._reactives.items():
            reactive._initialize_reactive(obj, name)

    @classmethod
    def _reset_object(cls, obj: object) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reset reactive structures on object (to avoid reference cycles).\n\n        Args:\n            obj: A reactive object.\n        '
        getattr(obj, '__watchers', {}).clear()
        getattr(obj, '__computes', []).clear()

    def __set_name__(self, owner: Type[MessageTarget], name: str) -> None:
        if False:
            print('Hello World!')
        public_compute = f'compute_{name}'
        private_compute = f'_compute_{name}'
        compute_name = private_compute if hasattr(owner, private_compute) else public_compute
        if hasattr(owner, compute_name):
            try:
                computes = getattr(owner, '__computes')
            except AttributeError:
                computes = []
                setattr(owner, '__computes', computes)
            computes.append(name)
        self.name = name
        self.internal_name = f'_reactive_{name}'
        self.compute_name = compute_name
        default = self._default
        setattr(owner, f'_default_{name}', default)

    def __get__(self, obj: Reactable, obj_type: type[object]) -> ReactiveType:
        if False:
            return 10
        internal_name = self.internal_name
        if not hasattr(obj, internal_name):
            self._initialize_reactive(obj, self.name)
        if hasattr(obj, self.compute_name):
            value: ReactiveType
            old_value = getattr(obj, internal_name)
            _rich_traceback_omit = True
            value = getattr(obj, self.compute_name)()
            setattr(obj, internal_name, value)
            self._check_watchers(obj, self.name, old_value)
            return value
        else:
            return getattr(obj, internal_name)

    def __set__(self, obj: Reactable, value: ReactiveType) -> None:
        if False:
            for i in range(10):
                print('nop')
        _rich_traceback_omit = True
        self._initialize_reactive(obj, self.name)
        if hasattr(obj, self.compute_name):
            raise AttributeError(f"Can't set {obj}.{self.name!r}; reactive attributes with a compute method are read-only")
        name = self.name
        current_value = getattr(obj, name)
        private_validate_function = getattr(obj, f'_validate_{name}', None)
        if callable(private_validate_function):
            value = private_validate_function(value)
        public_validate_function = getattr(obj, f'validate_{name}', None)
        if callable(public_validate_function):
            value = public_validate_function(value)
        if current_value != value or self._always_update:
            setattr(obj, self.internal_name, value)
            self._check_watchers(obj, name, current_value)
            if self._run_compute:
                self._compute(obj)
            if self._layout or self._repaint:
                obj.refresh(repaint=self._repaint, layout=self._layout)

    @classmethod
    def _check_watchers(cls, obj: Reactable, name: str, old_value: Any):
        if False:
            i = 10
            return i + 15
        'Check watchers, and call watch methods / computes\n\n        Args:\n            obj: The reactable object.\n            name: Attribute name.\n            old_value: The old (previous) value of the attribute.\n        '
        _rich_traceback_omit = True
        internal_name = f'_reactive_{name}'
        value = getattr(obj, internal_name)

        async def await_watcher(awaitable: Awaitable) -> None:
            """Coroutine to await an awaitable returned from a watcher"""
            _rich_traceback_omit = True
            await awaitable
            obj.post_message(events.Callback(callback=partial(Reactive._compute, obj)))

        def invoke_watcher(watcher_object: Reactable, watch_function: Callable, old_value: object, value: object) -> None:
            if False:
                return 10
            'Invoke a watch function.\n\n            Args:\n                watcher_object: The object watching for the changes.\n                watch_function: A watch function, which may be sync or async.\n                old_value: The old value of the attribute.\n                value: The new value of the attribute.\n            '
            _rich_traceback_omit = True
            param_count = count_parameters(watch_function)
            if param_count == 2:
                watch_result = watch_function(old_value, value)
            elif param_count == 1:
                watch_result = watch_function(value)
            else:
                watch_result = watch_function()
            if isawaitable(watch_result):
                watcher_object.call_next(partial(await_watcher, watch_result))
        private_watch_function = getattr(obj, f'_watch_{name}', None)
        if callable(private_watch_function):
            invoke_watcher(obj, private_watch_function, old_value, value)
        public_watch_function = getattr(obj, f'watch_{name}', None)
        if callable(public_watch_function):
            invoke_watcher(obj, public_watch_function, old_value, value)
        watchers: list[tuple[Reactable, Callable]]
        watchers = getattr(obj, '__watchers', {}).get(name, [])
        if watchers:
            watchers[:] = [(reactable, callback) for (reactable, callback) in watchers if reactable.is_attached and (not reactable._closing)]
            for (reactable, callback) in watchers:
                with reactable.prevent(*obj._prevent_message_types_stack[-1]):
                    invoke_watcher(reactable, callback, old_value, value)

    @classmethod
    def _compute(cls, obj: Reactable) -> None:
        if False:
            print('Hello World!')
        'Invoke all computes.\n\n        Args:\n            obj: Reactable object.\n        '
        _rich_traceback_guard = True
        for compute in obj._reactives.keys():
            try:
                compute_method = getattr(obj, f'compute_{compute}')
            except AttributeError:
                try:
                    compute_method = getattr(obj, f'_compute_{compute}')
                except AttributeError:
                    continue
            current_value = getattr(obj, f'_reactive_{compute}', getattr(obj, f'_default_{compute}', None))
            value = compute_method()
            setattr(obj, f'_reactive_{compute}', value)
            if value != current_value:
                cls._check_watchers(obj, compute, current_value)

class reactive(Reactive[ReactiveType]):
    """Create a reactive attribute.

    Args:
        default: A default value or callable that returns a default.
        layout: Perform a layout on change.
        repaint: Perform a repaint on change.
        init: Call watchers on initialize (post mount).
        always_update: Call watchers even when the new value equals the old value.
    """

    def __init__(self, default: ReactiveType | Callable[[], ReactiveType], *, layout: bool=False, repaint: bool=True, init: bool=True, always_update: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(default, layout=layout, repaint=repaint, init=init, always_update=always_update)

class var(Reactive[ReactiveType]):
    """Create a reactive attribute (with no auto-refresh).

    Args:
        default: A default value or callable that returns a default.
        init: Call watchers on initialize (post mount).
        always_update: Call watchers even when the new value equals the old value.
    """

    def __init__(self, default: ReactiveType | Callable[[], ReactiveType], init: bool=True, always_update: bool=False) -> None:
        if False:
            print('Hello World!')
        super().__init__(default, layout=False, repaint=False, init=init, always_update=always_update)

def _watch(node: DOMNode, obj: Reactable, attribute_name: str, callback: WatchCallbackType, *, init: bool=True) -> None:
    if False:
        print('Hello World!')
    'Watch a reactive variable on an object.\n\n    Args:\n        obj: The parent object.\n        attribute_name: The attribute to watch.\n        callback: A callable to call when the attribute changes.\n        init: True to call watcher initialization.\n    '
    if not hasattr(obj, '__watchers'):
        setattr(obj, '__watchers', {})
    watchers: dict[str, list[tuple[Reactable, Callable]]] = getattr(obj, '__watchers')
    watcher_list = watchers.setdefault(attribute_name, [])
    if callback in watcher_list:
        return
    watcher_list.append((node, callback))
    if init:
        current_value = getattr(obj, attribute_name, None)
        Reactive._check_watchers(obj, attribute_name, current_value)