class Solution(object):

    def minimumEffort(self, tasks):
        if False:
            return 10
        '\n        :type tasks: List[List[int]]\n        :rtype: int\n        '
        tasks.sort(key=lambda x: x[1] - x[0])
        result = 0
        for (a, m) in tasks:
            result = max(result + a, m)
        return result

class Solution2(object):

    def minimumEffort(self, tasks):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type tasks: List[List[int]]\n        :rtype: int\n        '
        tasks.sort(key=lambda x: x[0] - x[1])
        result = curr = 0
        for (a, m) in tasks:
            result += max(m - curr, 0)
            curr = max(curr, m) - a
        return result