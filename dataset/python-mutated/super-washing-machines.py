class Solution(object):

    def findMinMoves(self, machines):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type machines: List[int]\n        :rtype: int\n        '
        total = sum(machines)
        if total % len(machines):
            return -1
        (result, target, curr) = (0, total / len(machines), 0)
        for n in machines:
            curr += n - target
            result = max(result, max(n - target, abs(curr)))
        return result