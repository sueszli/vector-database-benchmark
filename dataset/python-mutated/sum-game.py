class Solution(object):

    def sumGame(self, num):
        if False:
            return 10
        '\n        :type num: str\n        :rtype: bool\n        '
        cnt = total = 0
        for i in xrange(len(num)):
            if num[i] == '?':
                cnt += -1 if i < len(num) // 2 else 1
            else:
                total += int(num[i]) if i < len(num) // 2 else -int(num[i])
        return True if cnt % 2 else total != cnt // 2 * 9