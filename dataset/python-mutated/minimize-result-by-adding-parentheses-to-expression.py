import itertools

class Solution(object):

    def minimizeResult(self, expression):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type expression: str\n        :rtype: str\n        '

        def stoi(s, i, j):
            if False:
                return 10
            result = 0
            for k in xrange(i, j):
                result = result * 10 + (ord(s[k]) - ord('0'))
            return result
        best = None
        min_val = float('inf')
        pos = expression.index('+')
        (left, right) = (stoi(expression, 0, pos), stoi(expression, pos + 1, len(expression)))
        (base1, base2_init) = (10 ** pos, 10 ** (len(expression) - (pos + 1) - 1))
        for i in xrange(pos):
            base2 = base2_init
            for j in xrange(pos + 1, len(expression)):
                (a, b) = divmod(left, base1)
                (c, d) = divmod(right, base2)
                val = max(a, 1) * (b + c) * max(d, 1)
                if val < min_val:
                    min_val = val
                    best = (i, j)
                base2 //= 10
            base1 //= 10
        return ''.join(itertools.chain((expression[i] for i in xrange(best[0])), '(', (expression[i] for i in xrange(best[0], best[1] + 1)), ')', (expression[i] for i in xrange(best[1] + 1, len(expression)))))

class Solution2(object):

    def minimizeResult(self, expression):
        if False:
            print('Hello World!')
        '\n        :type expression: str\n        :rtype: str\n        '
        best = None
        min_val = float('inf')
        pos = expression.index('+')
        (left, right) = (int(expression[0:pos]), int(expression[pos + 1:]))
        (base1, base2_init) = (10 ** pos, 10 ** (len(expression) - (pos + 1) - 1))
        for i in xrange(pos):
            base2 = base2_init
            for j in xrange(pos + 1, len(expression)):
                (a, b) = divmod(left, base1)
                (c, d) = divmod(right, base2)
                val = max(a, 1) * (b + c) * max(d, 1)
                if val < min_val:
                    min_val = val
                    best = (i, j)
                base2 //= 10
            base1 //= 10
        return ''.join([expression[:best[0]], '(', expression[best[0]:best[1] + 1], ')', expression[best[1] + 1:]])

class Solution3(object):

    def minimizeResult(self, expression):
        if False:
            return 10
        '\n        :type expression: str\n        :rtype: str\n        '
        best = None
        min_val = float('inf')
        pos = expression.index('+')
        for i in xrange(pos):
            for j in xrange(pos + 1, len(expression)):
                val = int(expression[:i] or '1') * (int(expression[i:pos]) + int(expression[pos + 1:j + 1])) * int(expression[j + 1:] or '1')
                if val < min_val:
                    min_val = val
                    best = (i, j)
        return ''.join([expression[:best[0]], '(', expression[best[0]:best[1] + 1], ')', expression[best[1] + 1:]])