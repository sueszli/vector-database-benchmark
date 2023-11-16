class Solution(object):

    def getHappyString(self, n, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        base = 2 ** (n - 1)
        if k > 3 * base:
            return ''
        result = [chr(ord('a') + (k - 1) // base)]
        while base > 1:
            k -= (k - 1) // base * base
            base //= 2
            result.append(('a' if result[-1] != 'a' else 'b') if (k - 1) // base == 0 else 'c' if result[-1] != 'c' else 'b')
        return ''.join(result)