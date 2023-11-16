from typing import Callable, Optional, TypeVar, Union
from reactivex import ConnectableObservable, Observable, abc
from reactivex import operators as ops
from reactivex import typing
from reactivex.subject import ReplaySubject
from reactivex.typing import Mapper
_TSource = TypeVar('_TSource')
_TResult = TypeVar('_TResult')

def replay_(mapper: Optional[Mapper[Observable[_TSource], Observable[_TResult]]]=None, buffer_size: Optional[int]=None, window: Optional[typing.RelativeTime]=None, scheduler: Optional[abc.SchedulerBase]=None) -> Callable[[Observable[_TSource]], Union[Observable[_TResult], ConnectableObservable[_TSource]]]:
    if False:
        print('Hello World!')
    'Returns an observable sequence that is the result of invoking the\n    mapper on a connectable observable sequence that shares a single\n    subscription to the underlying sequence replaying notifications\n    subject to a maximum time length for the replay buffer.\n\n    This operator is a specialization of Multicast using a\n    ReplaySubject.\n\n    Examples:\n        >>> res = replay(buffer_size=3)\n        >>> res = replay(buffer_size=3, window=500)\n        >>> res = replay(None, 3, 500)\n        >>> res = replay(lambda x: x.take(6).repeat(), 3, 500)\n\n    Args:\n        mapper: [Optional] Selector function which can use the multicasted\n            source sequence as many times as needed, without causing\n            multiple subscriptions to the source sequence. Subscribers to\n            the given source will receive all the notifications of the\n            source subject to the specified replay buffer trimming policy.\n        buffer_size: [Optional] Maximum element count of the replay\n            buffer.\n        window: [Optional] Maximum time length of the replay buffer.\n        scheduler: [Optional] Scheduler the observers are invoked on.\n\n    Returns:\n        An observable sequence that contains the elements of a\n    sequence produced by multicasting the source sequence within a\n    mapper function.\n    '
    if mapper:

        def subject_factory(scheduler: Optional[abc.SchedulerBase]=None) -> ReplaySubject[_TSource]:
            if False:
                i = 10
                return i + 15
            return ReplaySubject(buffer_size, window, scheduler)
        return ops.multicast(subject_factory=subject_factory, mapper=mapper)
    rs: ReplaySubject[_TSource] = ReplaySubject(buffer_size, window, scheduler)
    return ops.multicast(subject=rs)
__all__ = ['replay_']