import time
from math import inf
from .. import _core
from .._abc import Clock
from .._util import final
from ._run import GLOBAL_RUN_CONTEXT

@final
class MockClock(Clock):
    """A user-controllable clock suitable for writing tests.

    Args:
      rate (float): the initial :attr:`rate`.
      autojump_threshold (float): the initial :attr:`autojump_threshold`.

    .. attribute:: rate

       How many seconds of clock time pass per second of real time. Default is
       0.0, i.e. the clock only advances through manuals calls to :meth:`jump`
       or when the :attr:`autojump_threshold` is triggered. You can assign to
       this attribute to change it.

    .. attribute:: autojump_threshold

       The clock keeps an eye on the run loop, and if at any point it detects
       that all tasks have been blocked for this many real seconds (i.e.,
       according to the actual clock, not this clock), then the clock
       automatically jumps ahead to the run loop's next scheduled
       timeout. Default is :data:`math.inf`, i.e., to never autojump. You can
       assign to this attribute to change it.

       Basically the idea is that if you have code or tests that use sleeps
       and timeouts, you can use this to make it run much faster, totally
       automatically. (At least, as long as those sleeps/timeouts are
       happening inside Trio; if your test involves talking to external
       service and waiting for it to timeout then obviously we can't help you
       there.)

       You should set this to the smallest value that lets you reliably avoid
       "false alarms" where some I/O is in flight (e.g. between two halves of
       a socketpair) but the threshold gets triggered and time gets advanced
       anyway. This will depend on the details of your tests and test
       environment. If you aren't doing any I/O (like in our sleeping example
       above) then just set it to zero, and the clock will jump whenever all
       tasks are blocked.

       .. note:: If you use ``autojump_threshold`` and
          `wait_all_tasks_blocked` at the same time, then you might wonder how
          they interact, since they both cause things to happen after the run
          loop goes idle for some time. The answer is:
          `wait_all_tasks_blocked` takes priority. If there's a task blocked
          in `wait_all_tasks_blocked`, then the autojump feature treats that
          as active task and does *not* jump the clock.

    """

    def __init__(self, rate: float=0.0, autojump_threshold: float=inf):
        if False:
            for i in range(10):
                print('nop')
        self._real_base = 0.0
        self._virtual_base = 0.0
        self._rate = 0.0
        self._autojump_threshold = 0.0
        self._real_clock = time.perf_counter
        self.rate = rate
        self.autojump_threshold = autojump_threshold

    def __repr__(self) -> str:
        if False:
            return 10
        return '<MockClock, time={:.7f}, rate={} @ {:#x}>'.format(self.current_time(), self._rate, id(self))

    @property
    def rate(self) -> float:
        if False:
            print('Hello World!')
        return self._rate

    @rate.setter
    def rate(self, new_rate: float) -> None:
        if False:
            return 10
        if new_rate < 0:
            raise ValueError('rate must be >= 0')
        else:
            real = self._real_clock()
            virtual = self._real_to_virtual(real)
            self._virtual_base = virtual
            self._real_base = real
            self._rate = float(new_rate)

    @property
    def autojump_threshold(self) -> float:
        if False:
            print('Hello World!')
        return self._autojump_threshold

    @autojump_threshold.setter
    def autojump_threshold(self, new_autojump_threshold: float) -> None:
        if False:
            print('Hello World!')
        self._autojump_threshold = float(new_autojump_threshold)
        self._try_resync_autojump_threshold()

    def _try_resync_autojump_threshold(self) -> None:
        if False:
            while True:
                i = 10
        try:
            runner = GLOBAL_RUN_CONTEXT.runner
            if runner.is_guest:
                runner.force_guest_tick_asap()
        except AttributeError:
            pass
        else:
            runner.clock_autojump_threshold = self._autojump_threshold

    def _autojump(self) -> None:
        if False:
            return 10
        statistics = _core.current_statistics()
        jump = statistics.seconds_to_next_deadline
        if 0 < jump < inf:
            self.jump(jump)

    def _real_to_virtual(self, real: float) -> float:
        if False:
            return 10
        real_offset = real - self._real_base
        virtual_offset = self._rate * real_offset
        return self._virtual_base + virtual_offset

    def start_clock(self) -> None:
        if False:
            i = 10
            return i + 15
        self._try_resync_autojump_threshold()

    def current_time(self) -> float:
        if False:
            return 10
        return self._real_to_virtual(self._real_clock())

    def deadline_to_sleep_time(self, deadline: float) -> float:
        if False:
            return 10
        virtual_timeout = deadline - self.current_time()
        if virtual_timeout <= 0:
            return 0
        elif self._rate > 0:
            return virtual_timeout / self._rate
        else:
            return 999999999

    def jump(self, seconds: float) -> None:
        if False:
            while True:
                i = 10
        'Manually advance the clock by the given number of seconds.\n\n        Args:\n          seconds (float): the number of seconds to jump the clock forward.\n\n        Raises:\n          ValueError: if you try to pass a negative value for ``seconds``.\n\n        '
        if seconds < 0:
            raise ValueError("time can't go backwards")
        self._virtual_base += seconds