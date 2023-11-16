class Solution(object):

    def smallestNumber(self, pattern):
        if False:
            return 10
        '\n        :type pattern: str\n        :rtype: str\n        '
        result = []
        for i in xrange(len(pattern) + 1):
            if not (i == len(pattern) or pattern[i] == 'I'):
                continue
            for x in reversed(range(len(result) + 1, i + 1 + 1)):
                result.append(x)
        return ''.join(map(str, result))