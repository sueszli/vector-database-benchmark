import collections

class RecentCounter(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__q = collections.deque()

    def ping(self, t):
        if False:
            i = 10
            return i + 15
        '\n        :type t: int\n        :rtype: int\n        '
        self.__q.append(t)
        while self.__q[0] < t - 3000:
            self.__q.popleft()
        return len(self.__q)