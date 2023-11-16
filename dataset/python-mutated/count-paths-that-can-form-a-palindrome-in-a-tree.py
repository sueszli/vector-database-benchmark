import collections

class Solution(object):

    def countPalindromePaths(self, parent, s):
        if False:
            i = 10
            return i + 15
        '\n        :type parent: List[int]\n        :type s: str\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                while True:
                    i = 10
            result = 0
            cnt = collections.defaultdict(int)
            cnt[0] = 1
            stk = [(0, 0)]
            while stk:
                (u, mask) = stk.pop()
                if u:
                    mask ^= 1 << ord(s[u]) - ord('a')
                    result += cnt[mask] + sum((cnt[mask ^ 1 << i] if mask ^ 1 << i in cnt else 0 for i in xrange(26)))
                    cnt[mask] += 1
                for v in reversed(adj[u]):
                    stk.append((v, mask))
            return result
        adj = [[] for _ in xrange(len(parent))]
        for (u, p) in enumerate(parent):
            if p != -1:
                adj[p].append(u)
        return iter_dfs()
import collections

class Solution2(object):

    def countPalindromePaths(self, parent, s):
        if False:
            i = 10
            return i + 15
        '\n        :type parent: List[int]\n        :type s: str\n        :rtype: int\n        '

        def dfs(u, mask):
            if False:
                print('Hello World!')
            result = 0
            if u:
                mask ^= 1 << ord(s[u]) - ord('a')
                result += cnt[mask] + sum((cnt[mask ^ 1 << i] if mask ^ 1 << i in cnt else 0 for i in xrange(26)))
                cnt[mask] += 1
            return result + sum((dfs(v, mask) for v in adj[u]))
        adj = [[] for _ in xrange(len(parent))]
        for (u, p) in enumerate(parent):
            if p != -1:
                adj[p].append(u)
        cnt = collections.defaultdict(int)
        cnt[0] = 1
        return dfs(0, 0)