class Solution(object):

    def calPoints(self, ops):
        if False:
            return 10
        '\n        :type ops: List[str]\n        :rtype: int\n        '
        history = []
        for op in ops:
            if op == '+':
                history.append(history[-1] + history[-2])
            elif op == 'D':
                history.append(history[-1] * 2)
            elif op == 'C':
                history.pop()
            else:
                history.append(int(op))
        return sum(history)