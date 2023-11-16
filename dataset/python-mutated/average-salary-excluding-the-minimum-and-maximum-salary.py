class Solution(object):

    def average(self, salary):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type salary: List[int]\n        :rtype: float\n        '
        (total, mi, ma) = (0, float('inf'), float('-inf'))
        for s in salary:
            total += s
            (mi, ma) = (min(mi, s), max(ma, s))
        return 1.0 * (total - mi - ma) / (len(salary) - 2)

class Solution2(object):

    def average(self, salary):
        if False:
            print('Hello World!')
        '\n        :type salary: List[int]\n        :rtype: float\n        '
        return 1.0 * (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)