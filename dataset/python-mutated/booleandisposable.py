from threading import RLock
from reactivex.abc import DisposableBase

class BooleanDisposable(DisposableBase):
    """Represents a Disposable that can be checked for status."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        'Initializes a new instance of the BooleanDisposable class.'
        self.is_disposed = False
        self.lock = RLock()
        super().__init__()

    def dispose(self) -> None:
        if False:
            while True:
                i = 10
        'Sets the status to disposed'
        self.is_disposed = True