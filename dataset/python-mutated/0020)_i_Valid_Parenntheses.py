class Solution:

    def isValid(self, s: str) -> bool:
        if False:
            print('Hello World!')
        stack = []
        match = {'(': ')', '[': ']', '{': '}'}
        for c in s:
            if c in match:
                stack.append(c)
            elif not stack or match[stack.pop()] != c:
                return False
        return not stack