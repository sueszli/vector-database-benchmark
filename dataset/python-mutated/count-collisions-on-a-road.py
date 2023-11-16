class Solution(object):

    def countCollisions(self, directions):
        if False:
            return 10
        '\n        :type directions: str\n        :rtype: int\n        '
        result = cnt = 0
        smooth = 1
        for x in directions:
            if x == 'R':
                cnt += 1
            elif x == 'S' or (cnt or not smooth):
                result += cnt + int(x == 'L')
                cnt = smooth = 0
        return result