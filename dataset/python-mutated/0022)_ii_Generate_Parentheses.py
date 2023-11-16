class Solution:

    def generateParenthesis(self, n):
        if False:
            print('Hello World!')
        ans = ['()']
        for _ in range(2, n + 1):
            new = set()
            for s in ans:
                for i in range(len(s)):
                    a = s[:i] + '()' + s[i:]
                    new.add(a)
            ans = list(new)
        return ans