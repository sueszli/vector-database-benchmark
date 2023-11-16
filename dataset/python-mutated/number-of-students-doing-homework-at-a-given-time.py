import itertools

class Solution(object):

    def busyStudent(self, startTime, endTime, queryTime):
        if False:
            while True:
                i = 10
        '\n        :type startTime: List[int]\n        :type endTime: List[int]\n        :type queryTime: int\n        :rtype: int\n        '
        return sum((s <= queryTime <= e for (s, e) in itertools.izip(startTime, endTime)))