import itertools

class Solution(object):

    def minimumTime(self, jobs, workers):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type jobs: List[int]\n        :type workers: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                i = 10
                return i + 15
            return (a + (b - 1)) // b
        jobs.sort()
        workers.sort()
        return max((ceil_divide(j, w) for (j, w) in itertools.izip(jobs, workers)))