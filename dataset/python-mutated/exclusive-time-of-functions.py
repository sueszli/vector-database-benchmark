class Solution(object):

    def exclusiveTime(self, n, logs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type logs: List[str]\n        :rtype: List[int]\n        '
        result = [0] * n
        (stk, prev) = ([], 0)
        for log in logs:
            tokens = log.split(':')
            if tokens[1] == 'start':
                if stk:
                    result[stk[-1]] += int(tokens[2]) - prev
                stk.append(int(tokens[0]))
                prev = int(tokens[2])
            else:
                result[stk.pop()] += int(tokens[2]) - prev + 1
                prev = int(tokens[2]) + 1
        return result