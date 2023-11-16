from typing import Any, Optional
from reactivex import Observable, abc
from reactivex.disposable import Disposable

def never_() -> Observable[Any]:
    if False:
        i = 10
        return i + 15
    'Returns a non-terminating observable sequence, which can be used\n    to denote an infinite duration (e.g. when using reactive joins).\n\n    Returns:\n        An observable sequence whose observers will never get called.\n    '

    def subscribe(observer: abc.ObserverBase[Any], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
        if False:
            while True:
                i = 10
        return Disposable()
    return Observable(subscribe)
__all__ = ['never_']