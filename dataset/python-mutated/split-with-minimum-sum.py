class Solution(object):

    def splitNum(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: int\n        '
        sorted_num = ''.join(sorted(str(num)))
        return int(sorted_num[::2]) + int(sorted_num[1::2])