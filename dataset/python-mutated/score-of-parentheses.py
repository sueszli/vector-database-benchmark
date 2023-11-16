class Solution(object):

    def scoreOfParentheses(self, S):
        if False:
            return 10
        '\n        :type S: str\n        :rtype: int\n        '
        (result, depth) = (0, 0)
        for i in xrange(len(S)):
            if S[i] == '(':
                depth += 1
            else:
                depth -= 1
                if S[i - 1] == '(':
                    result += 2 ** depth
        return result

class Solution2(object):

    def scoreOfParentheses(self, S):
        if False:
            return 10
        '\n        :type S: str\n        :rtype: int\n        '
        stack = [0]
        for c in S:
            if c == '(':
                stack.append(0)
            else:
                last = stack.pop()
                stack[-1] += max(1, 2 * last)
        return stack[0]