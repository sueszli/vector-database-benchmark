import math

class Solution(object):

    def minimumPerimeter(self, neededApples):
        if False:
            print('Hello World!')
        '\n        :type neededApples: int\n        :rtype: int\n        '
        (a, b, c, d) = (4.0, 6.0, 2.0, float(-neededApples))
        p = (3 * a * c - b ** 2) / (3 * a ** 2)
        q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
        assert (q / 2) ** 2 + (p / 3) ** 3 > 0
        x = (-q / 2 + ((q / 2) ** 2 + (p / 3) ** 3) ** 0.5) ** (1.0 / 3) + (-q / 2 - ((q / 2) ** 2 + (p / 3) ** 3) ** 0.5) ** (1.0 / 3)
        return 8 * int(math.ceil(x - b / (3 * a)))

class Solution2(object):

    def minimumPerimeter(self, neededApples):
        if False:
            i = 10
            return i + 15
        '\n        :type neededApples: int\n        :rtype: int\n        '
        x = int((2 * neededApples) ** (1.0 / 3))
        x -= x % 2
        assert (x - 2) * (x - 1) * x < 2 * neededApples < (x + 2) ** 3
        x += 2
        if (x - 2) * (x - 1) * x < 2 * neededApples:
            x += 2
        return 8 * (x - 2) // 2

class Solution3(object):

    def minimumPerimeter(self, neededApples):
        if False:
            return 10
        '\n        :type neededApples: int\n        :rtype: int\n        '

        def check(neededApples, x):
            if False:
                i = 10
                return i + 15
            return r * (2 * r + 1) * (2 * r + 2) >= neededApples
        (left, right) = (1, int((neededApples / 4.0) ** (1.0 / 3)))
        while left <= right:
            mid = left + (right - left) // 2
            if check(neededApples, mid):
                right = mid - 1
            else:
                left = mid + 1
        return 8 * left