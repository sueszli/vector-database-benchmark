class Solution(object):

    def divideString(self, s, k, fill):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type k: int\n        :type fill: str\n        :rtype: List[str]\n        '
        return [s[i:i + k] + fill * (i + k - len(s)) for i in xrange(0, len(s), k)]