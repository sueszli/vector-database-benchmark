class Solution(object):

    def removeInvalidParentheses(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: List[str]\n        '

        def findMinRemove(s):
            if False:
                while True:
                    i = 10
            (left_removed, right_removed) = (0, 0)
            for c in s:
                if c == '(':
                    left_removed += 1
                elif c == ')':
                    if not left_removed:
                        right_removed += 1
                    else:
                        left_removed -= 1
            return (left_removed, right_removed)

        def isValid(s):
            if False:
                print('Hello World!')
            sum = 0
            for c in s:
                if c == '(':
                    sum += 1
                elif c == ')':
                    sum -= 1
                if sum < 0:
                    return False
            return sum == 0

        def removeInvalidParenthesesHelper(start, left_removed, right_removed):
            if False:
                i = 10
                return i + 15
            if left_removed == 0 and right_removed == 0:
                tmp = ''
                for (i, c) in enumerate(s):
                    if i not in removed:
                        tmp += c
                if isValid(tmp):
                    res.append(tmp)
                return
            for i in xrange(start, len(s)):
                if right_removed == 0 and left_removed > 0 and (s[i] == '('):
                    if i == start or s[i] != s[i - 1]:
                        removed[i] = True
                        removeInvalidParenthesesHelper(i + 1, left_removed - 1, right_removed)
                        del removed[i]
                elif right_removed > 0 and s[i] == ')':
                    if i == start or s[i] != s[i - 1]:
                        removed[i] = True
                        removeInvalidParenthesesHelper(i + 1, left_removed, right_removed - 1)
                        del removed[i]
        (res, removed) = ([], {})
        (left_removed, right_removed) = findMinRemove(s)
        removeInvalidParenthesesHelper(0, left_removed, right_removed)
        return res