class Solution(object):

    def maxProfitAssignment(self, difficulty, profit, worker):
        if False:
            return 10
        '\n        :type difficulty: List[int]\n        :type profit: List[int]\n        :type worker: List[int]\n        :rtype: int\n        '
        jobs = zip(difficulty, profit)
        jobs.sort()
        worker.sort()
        (result, i, max_profit) = (0, 0, 0)
        for ability in worker:
            while i < len(jobs) and jobs[i][0] <= ability:
                max_profit = max(max_profit, jobs[i][1])
                i += 1
            result += max_profit
        return result