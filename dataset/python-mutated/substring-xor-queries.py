class Solution(object):

    def substringXorQueries(self, s, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type queries: List[List[int]]\n        :rtype: List[List[int]]\n        '
        mx = max((a ^ b for (a, b) in queries))
        lookup = {}
        for i in xrange(len(s)):
            curr = 0
            for j in xrange(i, len(s)):
                curr = (curr << 1) + int(s[j])
                if curr > mx:
                    break
                if curr not in lookup:
                    lookup[curr] = [i, j]
                if s[i] == '0':
                    break
        return [lookup[a ^ b] if a ^ b in lookup else [-1, -1] for (a, b) in queries]