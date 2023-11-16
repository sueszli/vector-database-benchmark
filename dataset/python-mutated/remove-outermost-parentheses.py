class Solution(object):

    def removeOuterParentheses(self, S):
        if False:
            return 10
        '\n        :type S: str\n        :rtype: str\n        '
        deep = 1
        (result, cnt) = ([], 0)
        for c in S:
            if c == '(' and cnt >= deep:
                result.append(c)
            if c == ')' and cnt > deep:
                result.append(c)
            cnt += 1 if c == '(' else -1
        return ''.join(result)