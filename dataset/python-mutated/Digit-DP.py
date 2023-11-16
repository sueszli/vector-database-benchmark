class Solution:

    def digitDP(self, n: int) -> int:
        if False:
            i = 10
            return i + 15
        s = str(n)

        @cache
        def dfs(pos, state, isLimit, isNum):
            if False:
                while True:
                    i = 10
            if pos == len(s):
                return int(isNum)
            ans = 0
            if not isNum:
                ans = dfs(pos + 1, state, False, False)
            minX = 0 if isNum else 1
            maxX = int(s[pos]) if isLimit else 9
            for x in range(minX, maxX + 1):
                if state >> x & 1 == 0:
                    ans += dfs(pos + 1, state | 1 << x, isLimit and x == maxX, True)
            return ans
        return dfs(0, 0, True, False)