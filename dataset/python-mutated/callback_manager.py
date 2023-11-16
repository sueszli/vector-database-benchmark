""" Provides ``PropertyCallbackManager`` and ``EventCallbackManager``
mixin classes for adding ``on_change`` and ``on_event`` callback
interfaces to classes.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from collections import defaultdict
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union, cast
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from ..core.has_props import Setter
    from ..core.types import ID
    from ..document.document import Document
    from ..document.events import DocumentPatchedEvent
__all__ = ('EventCallbackManager', 'PropertyCallbackManager')
EventCallbackWithEvent: TypeAlias = Callable[[Event], None]
EventCallbackWithoutEvent: TypeAlias = Callable[[], None]
EventCallback: TypeAlias = Union[EventCallbackWithEvent, EventCallbackWithoutEvent]
PropertyCallback: TypeAlias = Callable[[str, Any, Any], None]

class EventCallbackManager:
    """ A mixin class to provide an interface for registering and
    triggering event callbacks on the Python side.

    """
    document: Document | None
    id: ID
    subscribed_events: set[str]
    _event_callbacks: dict[str, list[EventCallback]]

    def __init__(self, *args: Any, **kw: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kw)
        self._event_callbacks = defaultdict(list)

    def on_event(self, event: str | type[Event], *callbacks: EventCallback) -> None:
        if False:
            return 10
        ' Run callbacks when the specified event occurs on this Model\n\n        Not all Events are supported for all Models.\n        See specific Events in :ref:`bokeh.events` for more information on\n        which Models are able to trigger them.\n        '
        if not isinstance(event, str) and issubclass(event, Event):
            event = event.event_name
        for callback in callbacks:
            if _nargs(callback) != 0:
                _check_callback(callback, ('event',), what='Event callback')
            self._event_callbacks[event].append(callback)
        self.subscribed_events.add(event)

    def _trigger_event(self, event: ModelEvent) -> None:
        if False:
            while True:
                i = 10

        def invoke() -> None:
            if False:
                i = 10
                return i + 15
            for callback in self._event_callbacks.get(event.event_name, []):
                if event.model is not None and self.id == event.model.id:
                    if _nargs(callback) == 0:
                        cast(EventCallbackWithoutEvent, callback)()
                    else:
                        cast(EventCallbackWithEvent, callback)(event)
        if self.document is not None:
            from ..model import Model
            self.document.callbacks.notify_event(cast(Model, self), event, invoke)
        else:
            invoke()

    def _update_event_callbacks(self) -> None:
        if False:
            return 10
        if self.document is None:
            return
        for key in self._event_callbacks:
            from ..model import Model
            self.document.callbacks.subscribe(key, cast(Model, self))

class PropertyCallbackManager:
    """ A mixin class to provide an interface for registering and
    triggering callbacks.

    """
    document: Document | None
    _callbacks: dict[str, list[PropertyCallback]]

    def __init__(self, *args: Any, **kw: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kw)
        self._callbacks = {}

    def on_change(self, attr: str, *callbacks: PropertyCallback) -> None:
        if False:
            return 10
        ' Add a callback on this object to trigger when ``attr`` changes.\n\n        Args:\n            attr (str) : an attribute name on this object\n            callback (callable) : a callback function to register\n\n        Returns:\n            None\n\n        '
        if len(callbacks) == 0:
            raise ValueError('on_change takes an attribute name and one or more callbacks, got only one parameter')
        _callbacks = self._callbacks.setdefault(attr, [])
        for callback in callbacks:
            if callback in _callbacks:
                continue
            _check_callback(callback, ('attr', 'old', 'new'))
            _callbacks.append(callback)

    def remove_on_change(self, attr: str, *callbacks: PropertyCallback) -> None:
        if False:
            while True:
                i = 10
        ' Remove a callback from this object '
        if len(callbacks) == 0:
            raise ValueError('remove_on_change takes an attribute name and one or more callbacks, got only one parameter')
        _callbacks = self._callbacks.setdefault(attr, [])
        for callback in callbacks:
            _callbacks.remove(callback)

    def trigger(self, attr: str, old: Any, new: Any, hint: DocumentPatchedEvent | None=None, setter: Setter | None=None) -> None:
        if False:
            while True:
                i = 10
        ' Trigger callbacks for ``attr`` on this object.\n\n        Args:\n            attr (str) :\n            old (object) :\n            new (object) :\n\n        Returns:\n            None\n\n        '

        def invoke() -> None:
            if False:
                print('Hello World!')
            callbacks = self._callbacks.get(attr)
            if callbacks:
                for callback in callbacks:
                    callback(attr, old, new)
        if self.document is not None:
            from ..model import Model
            self.document.callbacks.notify_change(cast(Model, self), attr, old, new, hint, setter, invoke)
        else:
            invoke()

def _nargs(fn: Callable[..., Any]) -> int:
    if False:
        print('Hello World!')
    sig = signature(fn)
    (all_names, default_values) = get_param_info(sig)
    return len(all_names) - len(default_values)

def _check_callback(callback: Callable[..., Any], fargs: Sequence[str], what: str='Callback functions') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Bokeh-internal function to check callback signature'
    sig = signature(callback)
    formatted_args = str(sig)
    error_msg = what + ' must have signature func(%s), got func%s'
    (all_names, default_values) = get_param_info(sig)
    nargs = len(all_names) - len(default_values)
    if nargs != len(fargs):
        raise ValueError(error_msg % (', '.join(fargs), formatted_args))