class Solution(object):

    def distributeCandies(self, candies, num_people):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type candies: int\n        :type num_people: int\n        :rtype: List[int]\n        '
        p = int((2 * candies + 0.25) ** 0.5 - 0.5)
        remaining = candies - (p + 1) * p // 2
        (rows, cols) = divmod(p, num_people)
        result = [0] * num_people
        for i in xrange(num_people):
            result[i] = (i + 1) * (rows + 1) + rows * (rows + 1) // 2 * num_people if i < cols else (i + 1) * rows + (rows - 1) * rows // 2 * num_people
        result[cols] += remaining
        return result

class Solution2(object):

    def distributeCandies(self, candies, num_people):
        if False:
            print('Hello World!')
        '\n        :type candies: int\n        :type num_people: int\n        :rtype: List[int]\n        '
        (left, right) = (1, candies)
        while left <= right:
            mid = left + (right - left) // 2
            if not mid <= candies * 2 // (mid + 1):
                right = mid - 1
            else:
                left = mid + 1
        p = right
        remaining = candies - (p + 1) * p // 2
        (rows, cols) = divmod(p, num_people)
        result = [0] * num_people
        for i in xrange(num_people):
            result[i] = (i + 1) * (rows + 1) + rows * (rows + 1) // 2 * num_people if i < cols else (i + 1) * rows + (rows - 1) * rows // 2 * num_people
        result[cols] += remaining
        return result

class Solution3(object):

    def distributeCandies(self, candies, num_people):
        if False:
            while True:
                i = 10
        '\n        :type candies: int\n        :type num_people: int\n        :rtype: List[int]\n        '
        result = [0] * num_people
        i = 0
        while candies != 0:
            result[i % num_people] += min(candies, i + 1)
            candies -= min(candies, i + 1)
            i += 1
        return result