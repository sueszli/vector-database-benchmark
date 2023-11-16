class Solution(object):

    def thousandSeparator(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: str\n        '
        result = []
        s = str(n)
        for (i, c) in enumerate(str(n)):
            if i and (len(s) - i) % 3 == 0:
                result.append('.')
            result.append(c)
        return ''.join(result)