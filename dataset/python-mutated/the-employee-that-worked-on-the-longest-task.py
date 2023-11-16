class Solution(object):

    def hardestWorker(self, n, logs):
        if False:
            return 10
        '\n        :type n: int\n        :type logs: List[List[int]]\n        :rtype: int\n        '
        return logs[max(xrange(len(logs)), key=lambda x: (logs[x][1] - (logs[x - 1][1] if x - 1 >= 0 else 0), -logs[x][0]))][0]