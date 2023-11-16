import sys
import typing
from tenacity import BaseRetrying
from tenacity import DoAttempt
from tenacity import DoSleep
from tenacity import RetryCallState
from tornado import gen
if typing.TYPE_CHECKING:
    from tornado.concurrent import Future
_RetValT = typing.TypeVar('_RetValT')

class TornadoRetrying(BaseRetrying):

    def __init__(self, sleep: 'typing.Callable[[float], Future[None]]'=gen.sleep, **kwargs: typing.Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.sleep = sleep

    @gen.coroutine
    def __call__(self, fn: 'typing.Callable[..., typing.Union[typing.Generator[typing.Any, typing.Any, _RetValT], Future[_RetValT]]]', *args: typing.Any, **kwargs: typing.Any) -> 'typing.Generator[typing.Any, typing.Any, _RetValT]':
        if False:
            print('Hello World!')
        self.begin()
        retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        while True:
            do = self.iter(retry_state=retry_state)
            if isinstance(do, DoAttempt):
                try:
                    result = (yield fn(*args, **kwargs))
                except BaseException:
                    retry_state.set_exception(sys.exc_info())
                else:
                    retry_state.set_result(result)
            elif isinstance(do, DoSleep):
                retry_state.prepare_for_next_attempt()
                yield self.sleep(do)
            else:
                raise gen.Return(do)