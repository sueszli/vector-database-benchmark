from typing import Callable, Optional, TypeVar, Union
from reactivex import ConnectableObservable, Observable, abc, compose
from reactivex import operators as ops
from reactivex.subject import Subject
from reactivex.typing import Mapper
_TSource = TypeVar('_TSource')
_TResult = TypeVar('_TResult')

def publish_(mapper: Optional[Mapper[Observable[_TSource], Observable[_TResult]]]=None) -> Callable[[Observable[_TSource]], Union[Observable[_TResult], ConnectableObservable[_TSource]]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns an observable sequence that is the result of invoking the\n    mapper on a connectable observable sequence that shares a single\n    subscription to the underlying sequence. This operator is a\n    specialization of Multicast using a regular Subject.\n\n    Example:\n        >>> res = publish()\n        >>> res = publish(lambda x: x)\n\n    mapper: [Optional] Selector function which can use the\n        multicasted source sequence as many times as needed, without causing\n        multiple subscriptions to the source sequence. Subscribers to the\n        given source will receive all notifications of the source from the\n        time of the subscription on.\n\n    Returns:\n        An observable sequence that contains the elements of a sequence\n        produced by multicasting the source sequence within a mapper\n        function.\n    '
    if mapper:

        def factory(scheduler: Optional[abc.SchedulerBase]=None) -> Subject[_TSource]:
            if False:
                for i in range(10):
                    print('nop')
            return Subject()
        return ops.multicast(subject_factory=factory, mapper=mapper)
    subject: Subject[_TSource] = Subject()
    return ops.multicast(subject=subject)

def share_() -> Callable[[Observable[_TSource]], Observable[_TSource]]:
    if False:
        while True:
            i = 10
    'Share a single subscription among multple observers.\n\n    Returns a new Observable that multicasts (shares) the original\n    Observable. As long as there is at least one Subscriber this\n    Observable will be subscribed and emitting data. When all\n    subscribers have unsubscribed it will unsubscribe from the source\n    Observable.\n\n    This is an alias for a composed publish() and ref_count().\n    '
    return compose(ops.publish(), ops.ref_count())
__all__ = ['publish_', 'share_']