class TimeTrigger(object):
    """Trigger based on a fixed time interval.

    This trigger accepts iterations with a given interval time.

    Args:
        period (float): Interval time. It is given in seconds.

    """

    def __init__(self, period):
        if False:
            for i in range(10):
                print('nop')
        self._period = period
        self._next_time = self._period

    def __call__(self, trainer):
        if False:
            i = 10
            return i + 15
        if self._next_time < trainer.elapsed_time:
            self._next_time += self._period
            return True
        else:
            return False

    def serialize(self, serializer):
        if False:
            i = 10
            return i + 15
        self._next_time = serializer('next_time', self._next_time)