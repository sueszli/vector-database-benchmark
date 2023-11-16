class Solution(object):

    def minSessions(self, tasks, sessionTime):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type tasks: List[int]\n        :type sessionTime: int\n        :rtype: int\n        '
        dp = [float('inf') for _ in xrange(1 << len(tasks))]
        dp[0] = 0
        for mask in xrange(len(dp) - 1):
            basis = 1
            for task in tasks:
                new_mask = mask | basis
                basis <<= 1
                if new_mask == mask:
                    continue
                if dp[mask] % sessionTime + task > sessionTime:
                    task += sessionTime - dp[mask] % sessionTime
                dp[new_mask] = min(dp[new_mask], dp[mask] + task)
        return (dp[-1] + sessionTime - 1) // sessionTime

class Solution2(object):

    def minSessions(self, tasks, sessionTime):
        if False:
            while True:
                i = 10
        '\n        :type tasks: List[int]\n        :type sessionTime: int\n        :rtype: int\n        '
        dp = [[float('inf')] * 2 for _ in xrange(1 << len(tasks))]
        dp[0] = [0, sessionTime]
        for mask in xrange(len(dp) - 1):
            basis = 1
            for task in tasks:
                new_mask = mask | basis
                basis <<= 1
                if new_mask == mask:
                    continue
                if dp[mask][1] + task <= sessionTime:
                    dp[new_mask] = min(dp[new_mask], [dp[mask][0], dp[mask][1] + task])
                else:
                    dp[new_mask] = min(dp[new_mask], [dp[mask][0] + 1, task])
        return dp[-1][0]