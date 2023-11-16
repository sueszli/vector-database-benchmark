from enum import Enum, auto
from functools import partial
from typing import Callable, List, Optional, Union, overload
from sanic.base.meta import SanicMeta
from sanic.exceptions import BadRequest
from sanic.models.futures import FutureListener
from sanic.models.handler_types import ListenerType, Sanic

class ListenerEvent(str, Enum):

    def _generate_next_value_(name: str, *args) -> str:
        if False:
            return 10
        return name.lower()
    BEFORE_SERVER_START = 'server.init.before'
    AFTER_SERVER_START = 'server.init.after'
    BEFORE_SERVER_STOP = 'server.shutdown.before'
    AFTER_SERVER_STOP = 'server.shutdown.after'
    MAIN_PROCESS_START = auto()
    MAIN_PROCESS_READY = auto()
    MAIN_PROCESS_STOP = auto()
    RELOAD_PROCESS_START = auto()
    RELOAD_PROCESS_STOP = auto()
    BEFORE_RELOAD_TRIGGER = auto()
    AFTER_RELOAD_TRIGGER = auto()

class ListenerMixin(metaclass=SanicMeta):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        self._future_listeners: List[FutureListener] = []

    def _apply_listener(self, listener: FutureListener):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @overload
    def listener(self, listener_or_event: ListenerType[Sanic], event_or_none: str, apply: bool=...) -> ListenerType[Sanic]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def listener(self, listener_or_event: str, event_or_none: None=..., apply: bool=...) -> Callable[[ListenerType[Sanic]], ListenerType[Sanic]]:
        if False:
            print('Hello World!')
        ...

    def listener(self, listener_or_event: Union[ListenerType[Sanic], str], event_or_none: Optional[str]=None, apply: bool=True) -> Union[ListenerType[Sanic], Callable[[ListenerType[Sanic]], ListenerType[Sanic]]]:
        if False:
            while True:
                i = 10
        'Create a listener for a specific event in the application\'s lifecycle.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        .. note::\n            Overloaded signatures allow for different ways of calling this method, depending on the types of the arguments.\n\n            Usually, it is prederred to use one of the convenience methods such as `before_server_start` or `after_server_stop` instead of calling this method directly.\n\n            ```python\n            @app.before_server_start\n            async def prefered_method(_):\n                ...\n\n            @app.listener("before_server_start")\n            async def not_prefered_method(_):\n                ...\n\n        Args:\n            listener_or_event (Union[ListenerType[Sanic], str]): A listener function or an event name.\n            event_or_none (Optional[str]): The event name to listen for if `listener_or_event` is a function. Defaults to `None`.\n            apply (bool): Whether to apply the listener immediately. Defaults to `True`.\n\n        Returns:\n            Union[ListenerType[Sanic], Callable[[ListenerType[Sanic]], ListenerType[Sanic]]]: The listener or a callable that takes a listener.\n\n        Example:\n            The following code snippet shows how you can use this method as a decorator:\n\n            ```python\n            @bp.listener("before_server_start")\n            async def before_server_start(app, loop):\n                ...\n            ```\n        '

        def register_listener(listener: ListenerType[Sanic], event: str) -> ListenerType[Sanic]:
            if False:
                return 10
            'A helper function to register a listener for an event.\n\n            Typically will not be called directly.\n\n            Args:\n                listener (ListenerType[Sanic]): The listener function to\n                    register.\n                event (str): The event name to listen for.\n\n            Returns:\n                ListenerType[Sanic]: The listener function that was registered.\n            '
            nonlocal apply
            future_listener = FutureListener(listener, event)
            self._future_listeners.append(future_listener)
            if apply:
                self._apply_listener(future_listener)
            return listener
        if callable(listener_or_event):
            if event_or_none is None:
                raise BadRequest('Invalid event registration: Missing event name.')
            return register_listener(listener_or_event, event_or_none)
        else:
            return partial(register_listener, event=listener_or_event)

    def main_process_start(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            for i in range(10):
                print('nop')
        'Decorator for registering a listener for the main_process_start event.\n\n        This event is fired only on the main process and **NOT** on any\n        worker processes. You should typically use this event to initialize\n        resources that are shared across workers, or to initialize resources\n        that are not safe to be initialized in a worker process.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.main_process_start\n            async def on_main_process_start(app: Sanic):\n                print("Main process started")\n            ```\n        '
        return self.listener(listener, 'main_process_start')

    def main_process_ready(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            for i in range(10):
                print('nop')
        'Decorator for registering a listener for the main_process_ready event.\n\n        This event is fired only on the main process and **NOT** on any\n        worker processes. It is fired after the main process has started and\n        the Worker Manager has been initialized (ie, you will have access to\n        `app.manager` instance). The typical use case for this event is to\n        add a managed process to the Worker Manager.\n\n        See [Running custom processes](/en/guide/deployment/manager.html#running-custom-processes) and [Listeners](/en/guide/basics/listeners.html) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.main_process_ready\n            async def on_main_process_ready(app: Sanic):\n                print("Main process ready")\n            ```\n        '
        return self.listener(listener, 'main_process_ready')

    def main_process_stop(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            while True:
                i = 10
        'Decorator for registering a listener for the main_process_stop event.\n\n        This event is fired only on the main process and **NOT** on any\n        worker processes. You should typically use this event to clean up\n        resources that were initialized in the main_process_start event.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.main_process_stop\n            async def on_main_process_stop(app: Sanic):\n                print("Main process stopped")\n            ```\n        '
        return self.listener(listener, 'main_process_stop')

    def reload_process_start(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            return 10
        'Decorator for registering a listener for the reload_process_start event.\n\n        This event is fired only on the reload process and **NOT** on any\n        worker processes. This is similar to the main_process_start event,\n        except that it is fired only when the reload process is started.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.reload_process_start\n            async def on_reload_process_start(app: Sanic):\n                print("Reload process started")\n            ```\n        '
        return self.listener(listener, 'reload_process_start')

    def reload_process_stop(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            for i in range(10):
                print('nop')
        'Decorator for registering a listener for the reload_process_stop event.\n\n        This event is fired only on the reload process and **NOT** on any\n        worker processes. This is similar to the main_process_stop event,\n        except that it is fired only when the reload process is stopped.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.reload_process_stop\n            async def on_reload_process_stop(app: Sanic):\n                print("Reload process stopped")\n            ```\n        '
        return self.listener(listener, 'reload_process_stop')

    def before_reload_trigger(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            i = 10
            return i + 15
        'Decorator for registering a listener for the before_reload_trigger event.\n\n        This event is fired only on the reload process and **NOT** on any\n        worker processes. This event is fired before the reload process\n        triggers the reload. A change event has been detected and the reload\n        process is about to be triggered.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.before_reload_trigger\n            async def on_before_reload_trigger(app: Sanic):\n                print("Before reload trigger")\n            ```\n        '
        return self.listener(listener, 'before_reload_trigger')

    def after_reload_trigger(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            return 10
        'Decorator for registering a listener for the after_reload_trigger event.\n\n        This event is fired only on the reload process and **NOT** on any\n        worker processes. This event is fired after the reload process\n        triggers the reload. A change event has been detected and the reload\n        process has been triggered.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.after_reload_trigger\n            async def on_after_reload_trigger(app: Sanic, changed: set[str]):\n                print("After reload trigger, changed files: ", changed)\n            ```\n        '
        return self.listener(listener, 'after_reload_trigger')

    def before_server_start(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            for i in range(10):
                print('nop')
        'Decorator for registering a listener for the before_server_start event.\n\n        This event is fired on all worker processes. You should typically\n        use this event to initialize resources that are global in nature, or\n        will be shared across requests and various parts of the application.\n\n        A common use case for this event is to initialize a database connection\n        pool, or to initialize a cache client.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.before_server_start\n            async def on_before_server_start(app: Sanic):\n                print("Before server start")\n            ```\n        '
        return self.listener(listener, 'before_server_start')

    def after_server_start(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            i = 10
            return i + 15
        'Decorator for registering a listener for the after_server_start event.\n\n        This event is fired on all worker processes. You should typically\n        use this event to run background tasks, or perform other actions that\n        are not directly related to handling requests. In theory, it is\n        possible that some requests may be handled before this event is fired,\n        so you should not use this event to initialize resources that are\n        required for handling requests.\n\n        A common use case for this event is to start a background task that\n        periodically performs some action, such as clearing a cache or\n        performing a health check.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.after_server_start\n            async def on_after_server_start(app: Sanic):\n                print("After server start")\n            ```\n        '
        return self.listener(listener, 'after_server_start')

    def before_server_stop(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            for i in range(10):
                print('nop')
        'Decorator for registering a listener for the before_server_stop event.\n\n        This event is fired on all worker processes. This event is fired\n        before the server starts shutting down. You should not use this event\n        to perform any actions that are required for handling requests, as\n        some requests may continue to be handled after this event is fired.\n\n        A common use case for this event is to stop a background task that\n        was started in the after_server_start event.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.before_server_stop\n            async def on_before_server_stop(app: Sanic):\n                print("Before server stop")\n            ```\n        '
        return self.listener(listener, 'before_server_stop')

    def after_server_stop(self, listener: ListenerType[Sanic]) -> ListenerType[Sanic]:
        if False:
            for i in range(10):
                print('nop')
        'Decorator for registering a listener for the after_server_stop event.\n\n        This event is fired on all worker processes. This event is fired\n        after the server has stopped shutting down, and all requests have\n        been handled. You should typically use this event to clean up\n        resources that were initialized in the before_server_start event.\n\n        A common use case for this event is to close a database connection\n        pool, or to close a cache client.\n\n        See [Listeners](/en/guide/basics/listeners) for more details.\n\n        Args:\n            listener (ListenerType[Sanic]): The listener handler to attach.\n\n        Examples:\n            ```python\n            @app.after_server_stop\n            async def on_after_server_stop(app: Sanic):\n                print("After server stop")\n            ```\n        '
        return self.listener(listener, 'after_server_stop')