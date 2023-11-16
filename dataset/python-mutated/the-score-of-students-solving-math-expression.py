class Solution(object):

    def scoreOfStudents(self, s, answers):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type answers: List[int]\n        :rtype: int\n        '
        MAX_ANS = 1000
        n = (len(s) + 1) // 2
        dp = [[set() for _ in xrange(n)] for _ in xrange(n)]
        for i in xrange(n):
            dp[i][i].add(int(s[i * 2]))
        for l in xrange(1, n):
            for left in xrange(n - l):
                right = left + l
                for k in xrange(left, right):
                    if s[2 * k + 1] == '+':
                        dp[left][right].update((x + y for x in dp[left][k] for y in dp[k + 1][right] if x + y <= MAX_ANS))
                    else:
                        dp[left][right].update((x * y for x in dp[left][k] for y in dp[k + 1][right] if x * y <= MAX_ANS))
        target = eval(s)
        return sum((5 if ans == target else 2 if ans in dp[0][-1] else 0 for ans in answers))

class Solution2(object):

    def scoreOfStudents(self, s, answers):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type answers: List[int]\n        :rtype: int\n        '
        MAX_ANS = 1000

        def evaluate(s):
            if False:
                i = 10
                return i + 15

            def compute(operands, operators):
                if False:
                    for i in range(10):
                        print('nop')
                (right, left) = (operands.pop(), operands.pop())
                operands.append(ops[operators.pop()](left, right))
            ops = {'+': operator.add, '*': operator.mul}
            precedence = {'+': 0, '*': 1}
            (operands, operators, operand) = ([], [], 0)
            for c in s:
                if c.isdigit():
                    operands.append(int(c))
                else:
                    while operators and precedence[operators[-1]] >= precedence[c]:
                        compute(operands, operators)
                    operators.append(c)
            while operators:
                compute(operands, operators)
            return operands[-1]
        n = (len(s) + 1) // 2
        dp = [[set() for _ in xrange(n)] for _ in xrange(n)]
        for i in xrange(n):
            dp[i][i].add(int(s[i * 2]))
        for l in xrange(1, n):
            for left in xrange(n - l):
                right = left + l
                for k in xrange(left, right):
                    if s[2 * k + 1] == '+':
                        dp[left][right].update((x + y for x in dp[left][k] for y in dp[k + 1][right] if x + y <= MAX_ANS))
                    else:
                        dp[left][right].update((x * y for x in dp[left][k] for y in dp[k + 1][right] if x * y <= MAX_ANS))
        target = evaluate(s)
        return sum((5 if ans == target else 2 if ans in dp[0][-1] else 0 for ans in answers))