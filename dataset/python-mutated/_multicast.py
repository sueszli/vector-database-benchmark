from typing import Callable, Optional, TypeVar, Union
from reactivex import ConnectableObservable, Observable, abc
from reactivex import operators as ops
from reactivex.disposable import CompositeDisposable
_TSource = TypeVar('_TSource')
_TResult = TypeVar('_TResult')

def multicast_(subject: Optional[abc.SubjectBase[_TSource]]=None, *, subject_factory: Optional[Callable[[Optional[abc.SchedulerBase]], abc.SubjectBase[_TSource]]]=None, mapper: Optional[Callable[[Observable[_TSource]], Observable[_TResult]]]=None) -> Callable[[Observable[_TSource]], Union[Observable[_TResult], ConnectableObservable[_TSource]]]:
    if False:
        return 10
    'Multicasts the source sequence notifications through an\n    instantiated subject into all uses of the sequence within a mapper\n    function. Each subscription to the resulting sequence causes a\n    separate multicast invocation, exposing the sequence resulting from\n    the mapper function\'s invocation. For specializations with fixed\n    subject types, see Publish, PublishLast, and Replay.\n\n    Examples:\n        >>> res = multicast(observable)\n        >>> res = multicast(\n            subject_factory=lambda scheduler: Subject(),\n            mapper=lambda x: x\n        )\n\n    Args:\n        subject_factory: Factory function to create an intermediate\n            subject through which the source sequence\'s elements will be\n            multicast to the mapper function.\n        subject: Subject to push source elements into.\n        mapper: [Optional] Mapper function which can use the\n            multicasted source sequence subject to the policies enforced\n            by the created subject. Specified only if subject_factory"\n            is a factory function.\n\n    Returns:\n        An observable sequence that contains the elements of a sequence\n        produced by multicasting the source sequence within a mapper\n        function.\n    '

    def multicast(source: Observable[_TSource]) -> Union[Observable[_TResult], ConnectableObservable[_TSource]]:
        if False:
            for i in range(10):
                print('nop')
        if subject_factory:

            def subscribe(observer: abc.ObserverBase[_TResult], scheduler: Optional[abc.SchedulerBase]=None) -> abc.DisposableBase:
                if False:
                    return 10
                assert subject_factory
                connectable = source.pipe(ops.multicast(subject=subject_factory(scheduler)))
                assert mapper
                subscription = mapper(connectable).subscribe(observer, scheduler=scheduler)
                return CompositeDisposable(subscription, connectable.connect(scheduler))
            return Observable(subscribe)
        if not subject:
            raise ValueError('multicast: Subject cannot be None')
        ret: ConnectableObservable[_TSource] = ConnectableObservable(source, subject)
        return ret
    return multicast