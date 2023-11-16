import threading

class DiningPhilosophers(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._l = [threading.Lock() for _ in xrange(5)]

    def wantsToEat(self, philosopher, pickLeftFork, pickRightFork, eat, putLeftFork, putRightFork):
        if False:
            i = 10
            return i + 15
        '\n        :type philosopher: int\n        :type pickLeftFork: method\n        :type pickRightFork: method\n        :type eat: method\n        :type putLeftFork: method\n        :type putRightFork: method\n        :rtype: void\n        '
        (left, right) = (philosopher, (philosopher + 4) % 5)
        (first, second) = (left, right)
        if philosopher % 2 == 0:
            (first, second) = (left, right)
        else:
            (first, second) = (right, left)
        with self._l[first]:
            with self._l[second]:
                pickLeftFork()
                pickRightFork()
                eat()
                putLeftFork()
                putRightFork()