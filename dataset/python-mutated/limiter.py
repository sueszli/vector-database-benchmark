import logging
from types import FunctionType
from token_bucket import Limiter, MemoryStorage
logger = logging.getLogger(__name__)

class CallRateLimiter:
    KEY = b'default'

    def __init__(self, rate: int=6, capacity_factor: float=1.5, delay_factor: float=1.35) -> None:
        if False:
            while True:
                i = 10
        '\n        :param rate: Tokens to consume per time window (per sec)\n        :param capacity_factor: Used for capacity calculation, based on rate\n        :param delay_factor: Function call delay is multiplied by this value\n        on each next delayed call\n        '
        from twisted.internet import reactor
        self._reactor = reactor
        self._delay_factor = delay_factor
        self._limiter = Limiter(rate, capacity=int(capacity_factor * rate), storage=MemoryStorage())

    @property
    def delay_factor(self) -> float:
        if False:
            return 10
        return self._delay_factor

    def call(self, fn: FunctionType, *args, _limiter_key: bytes=KEY, _limiter_delay: float=1.0, **kwargs) -> None:
        if False:
            while True:
                i = 10
        "\n        Call the function if there are enough tokens in the bucket. Delay\n        the call otherwise.\n        :param fn: Function to call\n        :param args: Function's positional arguments\n        :param _limiter_key: Bucket key\n        :param _limiter_delay: Function call delay in seconds\n        :param kwargs: Function's keyword arguments\n        :return: None\n        "
        if self._limiter.consume(_limiter_key):
            fn(*args, **kwargs)
        else:
            logger.debug('Delaying function call by %r s: %r(%r, %r)', _limiter_delay, fn, args, kwargs)
            self._reactor.callLater(_limiter_delay, self.call, fn, *args, **kwargs, _limiter_key=_limiter_key, _limiter_delay=_limiter_delay * self._delay_factor)