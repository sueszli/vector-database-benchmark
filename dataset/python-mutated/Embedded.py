from robot.api.deco import keyword

class Embedded:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.called = 0

    @keyword('Called ${times} time(s)', types={'times': int})
    def called_times(self, times):
        if False:
            while True:
                i = 10
        self.called += 1
        if self.called != times:
            raise AssertionError('Called %s time(s), expected %s time(s).' % (self.called, times))