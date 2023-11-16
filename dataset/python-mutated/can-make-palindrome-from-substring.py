import itertools

class Solution(object):

    def canMakePaliQueries(self, s, queries):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type queries: List[List[int]]\n        :rtype: List[bool]\n        '
        CHARSET_SIZE = 26
        (curr, count) = ([0] * CHARSET_SIZE, [[0] * CHARSET_SIZE])
        for c in s:
            curr[ord(c) - ord('a')] += 1
            count.append(curr[:])
        return [sum(((b - a) % 2 for (a, b) in itertools.izip(count[left], count[right + 1]))) // 2 <= k for (left, right, k) in queries]