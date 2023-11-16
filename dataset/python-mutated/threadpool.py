import queue
import sys
from concurrent import futures
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, Optional, Set, TypeVar
_T = TypeVar('_T')

class ThreadPoolExecutor(futures.ThreadPoolExecutor):
    _max_workers: int

    def __init__(self, max_workers: Optional[int]=None, cancel_on_error: bool=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(max_workers=max_workers, **kwargs)
        self._cancel_on_error = cancel_on_error

    @property
    def max_workers(self) -> int:
        if False:
            return 10
        return self._max_workers

    def imap_unordered(self, fn: Callable[..., _T], *iterables: Iterable[Any]) -> Iterator[_T]:
        if False:
            return 10
        'Lazier version of map that does not preserve ordering of results.\n\n        It does not create all the futures at once to reduce memory usage.\n        '

        def create_taskset(n: int) -> Set[futures.Future]:
            if False:
                while True:
                    i = 10
            return {self.submit(fn, *args) for args in islice(it, n)}
        it = zip(*iterables)
        tasks = create_taskset(self.max_workers * 5)
        while tasks:
            (done, tasks) = futures.wait(tasks, return_when=futures.FIRST_COMPLETED)
            for fut in done:
                yield fut.result()
            tasks.update(create_taskset(len(done)))

    def shutdown(self, wait=True, *, cancel_futures=False):
        if False:
            return 10
        if sys.version_info > (3, 9):
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)
        else:
            with self._shutdown_lock:
                self._shutdown = True
                if cancel_futures:
                    while True:
                        try:
                            work_item = self._work_queue.get_nowait()
                        except queue.Empty:
                            break
                        if work_item is not None:
                            work_item.future.cancel()
                self._work_queue.put(None)
            if wait:
                for t in self._threads:
                    t.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        if self._cancel_on_error:
            self.shutdown(wait=True, cancel_futures=exc_val is not None)
        else:
            self.shutdown(wait=True)
        return False