import time
from abc import ABCMeta, abstractmethod
from typing import Union
from ..lock_helper import LockContext, LockContextType

class BaseTime(metaclass=ABCMeta):
    """
    Overview:
        Abstract time interface
    """

    @abstractmethod
    def time(self) -> Union[int, float]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get time information\n\n        Returns:\n            - time(:obj:`float, int`): time information\n        '
        raise NotImplementedError

class NaturalTime(BaseTime):
    """
    Overview:
        Natural time object

    Example:
        >>> from ding.utils.autolog.time_ctl import NaturalTime
        >>> time_ = NaturalTime()
    """

    def __init__(self):
        if False:
            return 10
        self.__last_time = None

    def time(self) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Get current natural time (float format, unix timestamp)\n\n        Returns:\n            - time(:obj:`float`): unix timestamp\n\n        Example:\n            >>> from ding.utils.autolog.time_ctl import NaturalTime\n            >>> time_ = NaturalTime()\n            >>> time_.time()\n            1603896383.8811457\n        '
        _current_time = time.time()
        if self.__last_time is not None:
            _current_time = max(_current_time, self.__last_time)
        self.__last_time = _current_time
        return _current_time

class TickTime(BaseTime):
    """
    Overview:
        Tick time object

    Example:
        >>> from ding.utils.autolog.time_ctl import TickTime
        >>> time_ = TickTime()
    """

    def __init__(self, init: int=0):
        if False:
            return 10
        '\n        Overview:\n            Constructor of TickTime\n\n        Args:\n            init (int, optional): init tick time, default is 1\n        '
        self.__tick_time = init

    def step(self, delta: int=1) -> int:
        if False:
            while True:
                i = 10
        '\n        Overview\n            Step the time forward for this TickTime\n\n        Args:\n             delta (int, optional): steps to step forward, default is 1\n\n        Returns:\n            int: new time after stepping\n\n        Example:\n            >>> from ding.utils.autolog.time_ctl import TickTime\n            >>> time_ = TickTime(0)\n            >>> time_.step()\n            1\n            >>> time_.step(2)\n            3\n        '
        if not isinstance(delta, int):
            raise TypeError('Delta should be positive int, but {actual} found.'.format(actual=type(delta).__name__))
        elif delta < 1:
            raise ValueError('Delta should be no less than 1, but {actual} found.'.format(actual=repr(delta)))
        else:
            self.__tick_time += delta
            return self.__tick_time

    def time(self) -> int:
        if False:
            return 10
        '\n        Overview\n            Get current tick time\n\n        Returns:\n            int: current tick time\n\n        Example:\n            >>> from ding.utils.autolog.time_ctl import TickTime\n            >>> time_ = TickTime(0)\n            >>> time_.step()\n            >>> time_.time()\n            1\n        '
        return self.__tick_time

class TimeProxy(BaseTime):
    """
    Overview:
        Proxy of time object, it can freeze time, sometimes useful when reproducing.
        This object is thread-safe, and also freeze and unfreeze operation is strictly ordered.

    Example:
        >>> from ding.utils.autolog.time_ctl import TickTime, TimeProxy
        >>> tick_time_ = TickTime()
        >>> time_ = TimeProxy(tick_time_)
        >>> tick_time_.step()
        >>> print(tick_time_.time(), time_.time(), time_.current_time())
        1 1 1
        >>> time_.freeze()
        >>> tick_time_.step()
        >>> print(tick_time_.time(), time_.time(), time_.current_time())
        2 1 2
        >>> time_.unfreeze()
        >>> print(tick_time_.time(), time_.time(), time_.current_time())
        2 2 2
    """

    def __init__(self, time_: BaseTime, frozen: bool=False, lock_type: LockContextType=LockContextType.THREAD_LOCK):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Constructor for Time proxy\n\n        Args:\n            time_ (BaseTime): another time object it based on\n            frozen (bool, optional): this object will be frozen immediately if true, otherwise not, default is False\n            lock_type (LockContextType, optional): type of the lock, default is THREAD_LOCK\n        '
        self.__time = time_
        self.__current_time = self.__time.time()
        self.__frozen = frozen
        self.__lock = LockContext(lock_type)
        self.__frozen_lock = LockContext(lock_type)
        if self.__frozen:
            self.__frozen_lock.acquire()

    @property
    def is_frozen(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get if this time proxy object is frozen\n\n        Returns:\n            bool: true if it is frozen, otherwise false\n        '
        with self.__lock:
            return self.__frozen

    def freeze(self):
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Freeze this time proxy\n        '
        with self.__lock:
            self.__frozen_lock.acquire()
            self.__frozen = True
            self.__current_time = self.__time.time()

    def unfreeze(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Unfreeze this time proxy\n        '
        with self.__lock:
            self.__frozen = False
            self.__frozen_lock.release()

    def time(self) -> Union[int, float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Get time (may be frozen time)\n\n        Returns:\n            int or float: the time\n        '
        with self.__lock:
            if self.__frozen:
                return self.__current_time
            else:
                return self.__time.time()

    def current_time(self) -> Union[int, float]:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Get current time (will not be frozen time)\n\n        Returns:\n            int or float: current time\n        '
        return self.__time.time()