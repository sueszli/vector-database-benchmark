class Solution(object):

    def validUtf8(self, data):
        if False:
            return 10
        '\n        :type data: List[int]\n        :rtype: bool\n        '
        count = 0
        for c in data:
            if count == 0:
                if c >> 5 == 6:
                    count = 1
                elif c >> 4 == 14:
                    count = 2
                elif c >> 3 == 30:
                    count = 3
                elif c >> 7:
                    return False
            else:
                if c >> 6 != 2:
                    return False
                count -= 1
        return count == 0