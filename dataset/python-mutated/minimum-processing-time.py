class Solution(object):

    def minProcessingTime(self, processorTime, tasks):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type processorTime: List[int]\n        :type tasks: List[int]\n        :rtype: int\n        '
        K = 4
        processorTime.sort()
        tasks.sort(reverse=True)
        result = 0
        for i in xrange(len(processorTime)):
            for j in xrange(K):
                result = max(result, processorTime[i] + tasks[i * K + j])
        return result