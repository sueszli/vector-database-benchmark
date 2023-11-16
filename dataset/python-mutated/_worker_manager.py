"""
A class to manage [workers](/guide/workers) for an app.

You access this object via [App.workers][textual.app.App.workers] or [Widget.workers][textual.dom.DOMNode.workers].
"""
from __future__ import annotations
import asyncio
from collections import Counter
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Iterable, Iterator
import rich.repr
from .worker import Worker, WorkerState, WorkType
if TYPE_CHECKING:
    from .app import App
    from .dom import DOMNode

@rich.repr.auto(angular=True)
class WorkerManager:
    """An object to manager a number of workers.

    You will not have to construct this class manually, as widgets, screens, and apps
    have a worker manager accessibly via a `workers` attribute.
    """

    def __init__(self, app: App) -> None:
        if False:
            return 10
        'Initialize a worker manager.\n\n        Args:\n            app: An App instance.\n        '
        self._app = app
        'A reference to the app.'
        self._workers: set[Worker] = set()
        'The workers being managed.'

    def __rich_repr__(self) -> rich.repr.Result:
        if False:
            for i in range(10):
                print('nop')
        counter: Counter[WorkerState] = Counter()
        counter.update((worker.state for worker in self._workers))
        for (state, count) in sorted(counter.items()):
            yield (state.name, count)

    def __iter__(self) -> Iterator[Worker[Any]]:
        if False:
            i = 10
            return i + 15
        return iter(sorted(self._workers, key=attrgetter('_created_time')))

    def __reversed__(self) -> Iterator[Worker[Any]]:
        if False:
            return 10
        return iter(sorted(self._workers, key=attrgetter('_created_time'), reverse=True))

    def __bool__(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self._workers)

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._workers)

    def __contains__(self, worker: object) -> bool:
        if False:
            print('Hello World!')
        return worker in self._workers

    def add_worker(self, worker: Worker, start: bool=True, exclusive: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        'Add a new worker.\n\n        Args:\n            worker: A Worker instance.\n            start: Start the worker if True, otherwise the worker must be started manually.\n            exclusive: Cancel all workers in the same group as `worker`.\n        '
        if exclusive and worker.group:
            self.cancel_group(worker.node, worker.group)
        self._workers.add(worker)
        if start:
            worker._start(self._app, self._remove_worker)

    def _new_worker(self, work: WorkType, node: DOMNode, *, name: str | None='', group: str='default', description: str='', exit_on_error: bool=True, start: bool=True, exclusive: bool=False, thread: bool=False) -> Worker:
        if False:
            print('Hello World!')
        'Create a worker from a function, coroutine, or awaitable.\n\n        Args:\n            work: A callable, a coroutine, or other awaitable.\n            name: A name to identify the worker.\n            group: The worker group.\n            description: A description of the worker.\n            exit_on_error: Exit the app if the worker raises an error. Set to `False` to suppress exceptions.\n            start: Automatically start the worker.\n            exclusive: Cancel all workers in the same group.\n            thread: Mark the worker as a thread worker.\n\n        Returns:\n            A Worker instance.\n        '
        worker: Worker[Any] = Worker(node, work, name=name or getattr(work, '__name__', '') or '', group=group, description=description or repr(work), exit_on_error=exit_on_error, thread=thread)
        self.add_worker(worker, start=start, exclusive=exclusive)
        return worker

    def _remove_worker(self, worker: Worker) -> None:
        if False:
            print('Hello World!')
        'Remove a worker from the manager.\n\n        Args:\n            worker: A Worker instance.\n        '
        self._workers.discard(worker)

    def start_all(self) -> None:
        if False:
            print('Hello World!')
        'Start all the workers.'
        for worker in self._workers:
            worker._start(self._app, self._remove_worker)

    def cancel_all(self) -> None:
        if False:
            while True:
                i = 10
        'Cancel all workers.'
        for worker in self._workers:
            worker.cancel()

    def cancel_group(self, node: DOMNode, group: str) -> list[Worker]:
        if False:
            while True:
                i = 10
        'Cancel a single group.\n\n        Args:\n            node: Worker DOM node.\n            group: A group name.\n\n        Returns:\n            A list of workers that were cancelled.\n        '
        workers = [worker for worker in self._workers if worker.group == group and worker.node == node]
        for worker in workers:
            worker.cancel()
        return workers

    def cancel_node(self, node: DOMNode) -> list[Worker]:
        if False:
            i = 10
            return i + 15
        'Cancel all workers associated with a given node\n\n        Args:\n            node: A DOM node (widget, screen, or App).\n\n        Returns:\n            List of cancelled workers.\n        '
        workers = [worker for worker in self._workers if worker.node == node]
        for worker in workers:
            worker.cancel()
        return workers

    async def wait_for_complete(self, workers: Iterable[Worker] | None=None) -> None:
        """Wait for workers to complete.

        Args:
            workers: An iterable of workers or None to wait for all workers in the manager.
        """
        await asyncio.gather(*[worker.wait() for worker in workers or self])