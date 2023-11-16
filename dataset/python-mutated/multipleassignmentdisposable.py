from threading import RLock
from typing import Optional
from reactivex.abc import DisposableBase

class MultipleAssignmentDisposable(DisposableBase):
    """Represents a disposable resource whose underlying disposable
    resource can be replaced by another disposable resource."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.current: Optional[DisposableBase] = None
        self.is_disposed = False
        self.lock = RLock()
        super().__init__()

    def get_disposable(self) -> Optional[DisposableBase]:
        if False:
            print('Hello World!')
        return self.current

    def set_disposable(self, value: DisposableBase) -> None:
        if False:
            i = 10
            return i + 15
        'If the MultipleAssignmentDisposable has already been\n        disposed, assignment to this property causes immediate disposal\n        of the given disposable object.'
        with self.lock:
            should_dispose = self.is_disposed
            if not should_dispose:
                self.current = value
        if should_dispose and value is not None:
            value.dispose()
    disposable = property(get_disposable, set_disposable)

    def dispose(self) -> None:
        if False:
            i = 10
            return i + 15
        'Disposes the underlying disposable as well as all future\n        replacements.'
        old = None
        with self.lock:
            if not self.is_disposed:
                self.is_disposed = True
                old = self.current
                self.current = None
        if old is not None:
            old.dispose()