class Solution(object):

    def parseTernary(self, expression):
        if False:
            return 10
        '\n        :type expression: str\n        :rtype: str\n        '
        if not expression:
            return ''
        stack = []
        for c in expression[::-1]:
            if stack and stack[-1] == '?':
                stack.pop()
                first = stack.pop()
                stack.pop()
                second = stack.pop()
                if c == 'T':
                    stack.append(first)
                else:
                    stack.append(second)
            else:
                stack.append(c)
        return str(stack[-1])