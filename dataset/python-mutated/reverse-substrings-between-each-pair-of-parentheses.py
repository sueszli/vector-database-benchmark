class Solution(object):

    def reverseParentheses(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: str\n        '
        (stk, lookup) = ([], {})
        for (i, c) in enumerate(s):
            if c == '(':
                stk.append(i)
            elif c == ')':
                j = stk.pop()
                (lookup[i], lookup[j]) = (j, i)
        result = []
        (i, d) = (0, 1)
        while i < len(s):
            if i in lookup:
                i = lookup[i]
                d *= -1
            else:
                result.append(s[i])
            i += d
        return ''.join(result)

class Solution2(object):

    def reverseParentheses(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '
        stk = [[]]
        for c in s:
            if c == '(':
                stk.append([])
            elif c == ')':
                end = stk.pop()
                end.reverse()
                stk[-1].extend(end)
            else:
                stk[-1].append(c)
        return ''.join(stk.pop())