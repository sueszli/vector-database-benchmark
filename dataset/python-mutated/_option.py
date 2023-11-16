from __future__ import annotations
import os
from logging import getLogger
from typing import Any, Callable, Generic, TypeVar, cast
from reactpy._warnings import warn
_O = TypeVar('_O')
logger = getLogger(__name__)
UNDEFINED = cast(Any, object())

class Option(Generic[_O]):
    """An option that can be set using an environment variable of the same name"""

    def __init__(self, name: str, default: _O=UNDEFINED, mutable: bool=True, parent: Option[_O] | None=None, validator: Callable[[Any], _O]=lambda x: cast(_O, x)) -> None:
        if False:
            while True:
                i = 10
        self._name = name
        self._mutable = mutable
        self._validator = validator
        self._subscribers: list[Callable[[_O], None]] = []
        if name in os.environ:
            self._current = validator(os.environ[name])
        if parent is not None:
            if not (parent.mutable and self.mutable):
                raise TypeError('Parent and child options must be mutable')
            self._default = parent.default
            parent.subscribe(self.set_current)
        elif default is not UNDEFINED:
            self._default = default
        else:
            raise TypeError('Must specify either a default or a parent option')
        logger.debug(f'{self._name}={self.current}')

    @property
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        'The name of this option (used to load environment variables)'
        return self._name

    @property
    def mutable(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether this option can be modified after being loaded'
        return self._mutable

    @property
    def default(self) -> _O:
        if False:
            for i in range(10):
                print('nop')
        "This option's default value"
        return self._default

    @property
    def current(self) -> _O:
        if False:
            i = 10
            return i + 15
        try:
            return self._current
        except AttributeError:
            return self._default

    @current.setter
    def current(self, new: _O) -> None:
        if False:
            i = 10
            return i + 15
        self.set_current(new)

    def subscribe(self, handler: Callable[[_O], None]) -> Callable[[_O], None]:
        if False:
            while True:
                i = 10
        'Register a callback that will be triggered when this option changes'
        if not self.mutable:
            msg = 'Immutable options cannot be subscribed to.'
            raise TypeError(msg)
        self._subscribers.append(handler)
        handler(self.current)
        return handler

    def is_set(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether this option has a value other than its default.'
        return hasattr(self, '_current')

    def set_current(self, new: Any) -> None:
        if False:
            print('Hello World!')
        'Set the value of this option\n\n        Raises a ``TypeError`` if this option is not :attr:`Option.mutable`.\n        '
        old = self.current
        if new is old:
            return None
        if not self._mutable:
            msg = f'{self} cannot be modified after initial load'
            raise TypeError(msg)
        try:
            new = self._current = self._validator(new)
        except ValueError as error:
            raise ValueError(f'Invalid value for {self._name}: {new!r}') from error
        logger.debug(f'{self._name}={self._current}')
        if new != old:
            for sub_func in self._subscribers:
                sub_func(new)

    def set_default(self, new: _O) -> _O:
        if False:
            i = 10
            return i + 15
        'Set the value of this option if not :meth:`Option.is_set`\n\n        Returns the current value (a la :meth:`dict.set_default`)\n        '
        if not self.is_set():
            self.set_current(new)
        return self._current

    def reload(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reload this option from its environment variable'
        self.set_current(os.environ.get(self._name, self._default))

    def unset(self) -> None:
        if False:
            while True:
                i = 10
        'Remove the current value, the default will be used until it is set again.'
        if not self._mutable:
            msg = f'{self} cannot be modified after initial load'
            raise TypeError(msg)
        old = self.current
        delattr(self, '_current')
        if self.current != old:
            for sub_func in self._subscribers:
                sub_func(self.current)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'Option({self._name}={self.current!r})'

class DeprecatedOption(Option[_O]):
    """An option that will warn when it is accessed"""

    def __init__(self, *args: Any, message: str, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._deprecation_message = message

    @Option.current.getter
    def current(self) -> _O:
        if False:
            i = 10
            return i + 15
        try:
            msg = self._deprecation_message
        except AttributeError:
            pass
        else:
            warn(msg, DeprecationWarning)
        return super().current