import asyncio
from time import sleep
from typing import Callable, List, Tuple
import pytest
from textual import work
from textual._work_decorator import WorkerDeclarationError
from textual.app import App
from textual.worker import Worker, WorkerState, WorkType

class WorkApp(App):
    worker: Worker

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.states: list[WorkerState] = []

    @work
    async def async_work(self) -> str:
        await asyncio.sleep(0.1)
        return 'foo'

    @work(thread=True)
    async def async_thread_work(self) -> str:
        await asyncio.sleep(0.1)
        return 'foo'

    @work(thread=True)
    def thread_work(self) -> str:
        if False:
            print('Hello World!')
        sleep(0.1)
        return 'foo'

    def launch(self, worker) -> None:
        if False:
            i = 10
            return i + 15
        self.worker = worker()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if False:
            print('Hello World!')
        self.states.append(event.state)

async def work_with(launcher: Callable[[WorkApp], WorkType]) -> None:
    """Core code for testing a work decorator."""
    app = WorkApp()
    async with app.run_test() as pilot:
        app.launch(launcher(app))
        await app.workers.wait_for_complete()
        result = await app.worker.wait()
        assert result == 'foo'
        await pilot.pause()
        assert app.states == [WorkerState.PENDING, WorkerState.RUNNING, WorkerState.SUCCESS]

async def test_async_work() -> None:
    """It should be possible to decorate an async method as an async worker."""
    await work_with(lambda app: app.async_work)

async def test_async_thread_work() -> None:
    """It should be possible to decorate an async method as a thread worker."""
    await work_with(lambda app: app.async_thread_work)

async def test_thread_work() -> None:
    """It should be possible to decorate a non-async method as a thread worker."""
    await work_with(lambda app: app.thread_work)

def test_decorate_non_async_no_thread_argument() -> None:
    if False:
        i = 10
        return i + 15
    "Decorating a non-async method without saying explicitly that it's a thread is an error."
    with pytest.raises(WorkerDeclarationError):

        class _(App[None]):

            @work
            def foo(self) -> None:
                if False:
                    while True:
                        i = 10
                pass

def test_decorate_non_async_no_thread_is_false() -> None:
    if False:
        while True:
            i = 10
    "Decorating a non-async method and saying it isn't a thread is an error."
    with pytest.raises(WorkerDeclarationError):

        class _(App[None]):

            @work(thread=False)
            def foo(self) -> None:
                if False:
                    i = 10
                    return i + 15
                pass

class NestedWorkersApp(App[None]):

    def __init__(self, call_stack: List[str]):
        if False:
            for i in range(10):
                print('nop')
        self.call_stack = call_stack
        super().__init__()

    def call_from_stack(self):
        if False:
            for i in range(10):
                print('nop')
        if self.call_stack:
            call_now = self.call_stack.pop()
            getattr(self, call_now)()

    @work(thread=False)
    async def async_no_thread(self):
        self.call_from_stack()

    @work(thread=True)
    async def async_thread(self):
        self.call_from_stack()

    @work(thread=True)
    def thread(self):
        if False:
            i = 10
            return i + 15
        self.call_from_stack()

@pytest.mark.parametrize('call_stack', [('async_no_thread', 'async_no_thread', 'async_no_thread'), ('async_no_thread', 'async_no_thread', 'async_thread'), ('async_no_thread', 'async_no_thread', 'thread'), ('async_no_thread', 'async_thread', 'async_no_thread'), ('async_no_thread', 'async_thread', 'async_thread'), ('async_no_thread', 'async_thread', 'thread'), ('async_no_thread', 'thread', 'async_no_thread'), ('async_no_thread', 'thread', 'async_thread'), ('async_no_thread', 'thread', 'thread'), ('async_thread', 'async_no_thread', 'async_no_thread'), ('async_thread', 'async_no_thread', 'async_thread'), ('async_thread', 'async_no_thread', 'thread'), ('async_thread', 'async_thread', 'async_no_thread'), ('async_thread', 'async_thread', 'async_thread'), ('async_thread', 'async_thread', 'thread'), ('async_thread', 'thread', 'async_no_thread'), ('async_thread', 'thread', 'async_thread'), ('async_thread', 'thread', 'thread'), ('thread', 'async_no_thread', 'async_no_thread'), ('thread', 'async_no_thread', 'async_thread'), ('thread', 'async_no_thread', 'thread'), ('thread', 'async_thread', 'async_no_thread'), ('thread', 'async_thread', 'async_thread'), ('thread', 'async_thread', 'thread'), ('thread', 'thread', 'async_no_thread'), ('thread', 'thread', 'async_thread'), ('thread', 'thread', 'thread'), ('async_no_thread', 'async_no_thread', 'thread', 'thread', 'async_thread', 'async_thread', 'async_no_thread', 'async_thread', 'async_no_thread', 'async_thread', 'thread', 'async_thread', 'async_thread', 'async_no_thread', 'async_no_thread', 'thread', 'thread', 'async_no_thread', 'async_no_thread', 'thread', 'async_no_thread', 'thread', 'thread')])
async def test_calling_workers_from_within_workers(call_stack: Tuple[str]):
    """Regression test for https://github.com/Textualize/textual/issues/3472.

    This makes sure we can nest worker calls without a problem.
    """
    app = NestedWorkersApp(list(call_stack))
    async with app.run_test():
        app.call_from_stack()
        for _ in range(len(call_stack)):
            await app.workers.wait_for_complete()
        assert app.call_stack == []