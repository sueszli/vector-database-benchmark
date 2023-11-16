class Solution(object):

    def lastRemaining(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        (start, step, direction) = (1, 2, 1)
        while n > 1:
            start += direction * (step * (n // 2) - step // 2)
            n //= 2
            step *= 2
            direction *= -1
        return start