import asyncio
import logging
from concurrent.futures import Future
from typing import List, Optional, TypeVar
from reactivex import abc, typing
from reactivex.disposable import CompositeDisposable, Disposable, SingleAssignmentDisposable
from .asyncioscheduler import AsyncIOScheduler
_TState = TypeVar('_TState')
log = logging.getLogger('Rx')

class AsyncIOThreadSafeScheduler(AsyncIOScheduler):
    """A scheduler that schedules work via the asyncio mainloop. This is a
    subclass of AsyncIOScheduler which uses the threadsafe asyncio methods.
    """

    def schedule(self, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            i = 10
            return i + 15
        'Schedules an action to be executed.\n\n        Args:\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                while True:
                    i = 10
            sad.disposable = self.invoke_action(action, state=state)
        handle = self._loop.call_soon_threadsafe(interval)

        def dispose() -> None:
            if False:
                while True:
                    i = 10
            if self._on_self_loop_or_not_running():
                handle.cancel()
                return
            future: 'Future[int]' = Future()

            def cancel_handle() -> None:
                if False:
                    return 10
                handle.cancel()
                future.set_result(0)
            self._loop.call_soon_threadsafe(cancel_handle)
            future.result()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_relative(self, duetime: typing.RelativeTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            for i in range(10):
                print('nop')
        'Schedules an action to be executed after duetime.\n\n        Args:\n            duetime: Relative time after which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        seconds = self.to_seconds(duetime)
        if seconds <= 0:
            return self.schedule(action, state=state)
        sad = SingleAssignmentDisposable()

        def interval() -> None:
            if False:
                while True:
                    i = 10
            sad.disposable = self.invoke_action(action, state=state)
        handle: List[asyncio.Handle] = []

        def stage2() -> None:
            if False:
                return 10
            handle.append(self._loop.call_later(seconds, interval))
        handle.append(self._loop.call_soon_threadsafe(stage2))

        def dispose() -> None:
            if False:
                i = 10
                return i + 15

            def do_cancel_handles() -> None:
                if False:
                    while True:
                        i = 10
                try:
                    handle.pop().cancel()
                    handle.pop().cancel()
                except Exception:
                    pass
            if self._on_self_loop_or_not_running():
                do_cancel_handles()
                return
            future: 'Future[int]' = Future()

            def cancel_handle() -> None:
                if False:
                    while True:
                        i = 10
                do_cancel_handles()
                future.set_result(0)
            self._loop.call_soon_threadsafe(cancel_handle)
            future.result()
        return CompositeDisposable(sad, Disposable(dispose))

    def schedule_absolute(self, duetime: typing.AbsoluteTime, action: typing.ScheduledAction[_TState], state: Optional[_TState]=None) -> abc.DisposableBase:
        if False:
            print('Hello World!')
        'Schedules an action to be executed at duetime.\n\n        Args:\n            duetime: Absolute time at which to execute the action.\n            action: Action to be executed.\n            state: [Optional] state to be given to the action function.\n\n        Returns:\n            The disposable object used to cancel the scheduled action\n            (best effort).\n        '
        duetime = self.to_datetime(duetime)
        return self.schedule_relative(duetime - self.now, action, state=state)

    def _on_self_loop_or_not_running(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns True if either self._loop is not running, or we're currently\n        executing on self._loop. In both cases, waiting for a future to be\n        resolved on the loop would result in a deadlock.\n        "
        if not self._loop.is_running():
            return True
        current_loop = None
        try:
            current_loop = asyncio.get_event_loop()
        except RuntimeError:
            pass
        return self._loop == current_loop