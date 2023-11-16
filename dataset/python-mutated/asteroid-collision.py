class Solution(object):

    def asteroidCollision(self, asteroids):
        if False:
            return 10
        '\n        :type asteroids: List[int]\n        :rtype: List[int]\n        '
        result = []
        for x in asteroids:
            if x > 0:
                result.append(x)
                continue
            while result and 0 < result[-1] < -x:
                result.pop()
            if result and 0 < result[-1]:
                if result[-1] == -x:
                    result.pop()
                continue
            result.append(x)
        return result

class Solution2(object):

    def asteroidCollision(self, asteroids):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type asteroids: List[int]\n        :rtype: List[int]\n        '
        result = []
        for x in asteroids:
            while result and x < 0 < result[-1]:
                if result[-1] < -x:
                    result.pop()
                    continue
                elif result[-1] == -x:
                    result.pop()
                break
            else:
                result.append(x)
        return result