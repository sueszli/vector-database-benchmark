class Solution(object):

    def largestGoodInteger(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: str\n        :rtype: str\n        '
        result = ''
        cnt = 0
        for (i, x) in enumerate(num):
            cnt += 1
            if i + 1 < len(num) and num[i] == num[i + 1]:
                continue
            if cnt >= 3:
                result = max(result, num[i])
            cnt = 0
        return result * 3

class Solution2(object):

    def largestGoodInteger(self, num):
        if False:
            return 10
        '\n        :type num: str\n        :rtype: str\n        '
        return max((num[i] if num[i] == num[i + 1] == num[i + 2] else '' for i in xrange(len(num) - 2))) * 3