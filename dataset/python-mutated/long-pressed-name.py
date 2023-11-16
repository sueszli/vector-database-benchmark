class Solution(object):

    def isLongPressedName(self, name, typed):
        if False:
            i = 10
            return i + 15
        '\n        :type name: str\n        :type typed: str\n        :rtype: bool\n        '
        i = 0
        for j in xrange(len(typed)):
            if i < len(name) and name[i] == typed[j]:
                i += 1
            elif j == 0 or typed[j] != typed[j - 1]:
                return False
        return i == len(name)