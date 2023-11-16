import collections
import bisect

class SnapshotArray(object):

    def __init__(self, length):
        if False:
            return 10
        '\n        :type length: int\n        '
        self.__A = collections.defaultdict(lambda : [[0, 0]])
        self.__snap_id = 0

    def set(self, index, val):
        if False:
            while True:
                i = 10
        '\n        :type index: int\n        :type val: int\n        :rtype: None\n        '
        if self.__A[index][-1][0] == self.__snap_id:
            self.__A[index][-1][1] = val
        else:
            self.__A[index].append([self.__snap_id, val])

    def snap(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        self.__snap_id += 1
        return self.__snap_id - 1

    def get(self, index, snap_id):
        if False:
            print('Hello World!')
        '\n        :type index: int\n        :type snap_id: int\n        :rtype: int\n        '
        i = bisect.bisect_left(self.__A[index], [snap_id + 1, float('-inf')]) - 1
        return self.__A[index][i][1]