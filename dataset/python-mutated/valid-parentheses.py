class Solution(object):

    def isValid(self, s):
        if False:
            return 10
        (stack, lookup) = ([], {'(': ')', '{': '}', '[': ']'})
        for parenthese in s:
            if parenthese in lookup:
                stack.append(parenthese)
            elif len(stack) == 0 or lookup[stack.pop()] != parenthese:
                return False
        return len(stack) == 0