class Solution(object):

    def isValid(self, S):
        if False:
            while True:
                i = 10
        '\n        :type S: str\n        :rtype: bool\n        '
        stack = []
        for i in S:
            if i == 'c':
                if stack[-2:] == ['a', 'b']:
                    stack.pop()
                    stack.pop()
                else:
                    return False
            else:
                stack.append(i)
        return not stack