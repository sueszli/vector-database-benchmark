class Solution(object):

    def sumOfMultiples(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '

        def f(d):
            if False:
                print('Hello World!')
            return d * ((1 + n // d) * (n // d) // 2)
        return f(3) + f(5) + f(7) - (f(3 * 5) + f(5 * 7) + f(7 * 3)) + f(3 * 5 * 7)