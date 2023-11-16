import itertools

class Solution(object):

    def printVertically(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: List[str]\n        '
        return [''.join(c).rstrip() for c in itertools.izip_longest(*s.split(), fillvalue=' ')]