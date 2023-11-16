class Solution(object):

    def digitsCount(self, d, low, high):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type d: int\n        :type low: int\n        :type high: int\n        :rtype: int\n        '

        def digitsCount(n, k):
            if False:
                return 10
            (pivot, result) = (1, 0)
            while n >= pivot:
                result += n // (10 * pivot) * pivot + min(pivot, max(n % (10 * pivot) - k * pivot + 1, 0))
                if k == 0:
                    result -= pivot
                pivot *= 10
            return result + 1
        return digitsCount(high, d) - digitsCount(low - 1, d)