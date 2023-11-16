from threading import RLock
from typing import Any, List
from reactivex import abc

class CompositeDisposable(abc.DisposableBase):
    """Represents a group of disposable resources that are disposed
    together"""

    def __init__(self, *args: Any):
        if False:
            for i in range(10):
                print('nop')
        if args and isinstance(args[0], list):
            self.disposable: List[abc.DisposableBase] = args[0]
        else:
            self.disposable = list(args)
        self.is_disposed = False
        self.lock = RLock()
        super(CompositeDisposable, self).__init__()

    def add(self, item: abc.DisposableBase) -> None:
        if False:
            return 10
        'Adds a disposable to the CompositeDisposable or disposes the\n        disposable if the CompositeDisposable is disposed\n\n        Args:\n            item: Disposable to add.'
        should_dispose = False
        with self.lock:
            if self.is_disposed:
                should_dispose = True
            else:
                self.disposable.append(item)
        if should_dispose:
            item.dispose()

    def remove(self, item: abc.DisposableBase) -> bool:
        if False:
            while True:
                i = 10
        'Removes and disposes the first occurrence of a disposable\n        from the CompositeDisposable.'
        if self.is_disposed:
            return False
        should_dispose = False
        with self.lock:
            if item in self.disposable:
                self.disposable.remove(item)
                should_dispose = True
        if should_dispose:
            item.dispose()
        return should_dispose

    def dispose(self) -> None:
        if False:
            print('Hello World!')
        'Disposes all disposable in the group and removes them from\n        the group.'
        if self.is_disposed:
            return
        with self.lock:
            self.is_disposed = True
            current_disposable = self.disposable
            self.disposable = []
        for disp in current_disposable:
            disp.dispose()

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Removes and disposes all disposable from the\n        CompositeDisposable, but does not dispose the\n        CompositeDisposable.'
        with self.lock:
            current_disposable = self.disposable
            self.disposable = []
        for disposable in current_disposable:
            disposable.dispose()

    def contains(self, item: abc.DisposableBase) -> bool:
        if False:
            i = 10
            return i + 15
        'Determines whether the CompositeDisposable contains a specific\n        disposable.\n\n        Args:\n            item: Disposable to search for\n\n        Returns:\n            True if the disposable was found; otherwise, False'
        return item in self.disposable

    def to_list(self) -> List[abc.DisposableBase]:
        if False:
            return 10
        return self.disposable[:]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self.disposable)

    @property
    def length(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.disposable)