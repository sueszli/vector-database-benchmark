class Solution(object):

    def productQueries(self, n, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        MOD = 10 ** 9 + 7
        prefix = [0]
        i = 0
        while 1 << i <= n:
            if n & 1 << i:
                prefix.append(prefix[-1] + i)
            i += 1
        return [pow(2, prefix[r + 1] - prefix[l], MOD) for (l, r) in queries]