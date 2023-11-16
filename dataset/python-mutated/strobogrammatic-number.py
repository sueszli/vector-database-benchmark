class Solution(object):
    lookup = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}

    def isStrobogrammatic(self, num):
        if False:
            i = 10
            return i + 15
        n = len(num)
        for i in xrange((n + 1) / 2):
            if num[n - 1 - i] not in self.lookup or num[i] != self.lookup[num[n - 1 - i]]:
                return False
        return True