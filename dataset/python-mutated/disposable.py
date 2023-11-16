from threading import RLock
from typing import Optional
from reactivex import typing
from reactivex.abc import DisposableBase
from reactivex.internal import noop
from reactivex.typing import Action

class Disposable(DisposableBase):
    """Main disposable class"""

    def __init__(self, action: Optional[typing.Action]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Creates a disposable object that invokes the specified\n        action when disposed.\n\n        Args:\n            action: Action to run during the first call to dispose.\n                The action is guaranteed to be run at most once.\n\n        Returns:\n            The disposable object that runs the given action upon\n            disposal.\n        '
        self.is_disposed = False
        self.action: Action = action or noop
        self.lock = RLock()
        super().__init__()

    def dispose(self) -> None:
        if False:
            i = 10
            return i + 15
        'Performs the task of cleaning up resources.'
        dispose = False
        with self.lock:
            if not self.is_disposed:
                dispose = True
                self.is_disposed = True
        if dispose:
            self.action()