from time import time

class DeltaProfiler:
    """
    This is a Python specific ProfileTimer.cxx.
    It's not related directly to the ProfileTimer code, it just
    shares some concepts.
    """

    def __init__(self, name=''):
        if False:
            print('Hello World!')
        self.name = name
        self.priorLabel = ''
        self.priorTime = 0
        self.active = 0

    def printDeltaTime(self, label):
        if False:
            for i in range(10):
                print('nop')
        if self.active:
            deltaTime = time() - self.priorTime
            print('%s DeltaTime %-25s to %-25s: %3.5f' % (self.name, self.priorLabel, label, deltaTime))
            self.priorLabel = label
            self.priorTime = time()