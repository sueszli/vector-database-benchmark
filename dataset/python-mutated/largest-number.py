class Solution(object):

    def largestNumber(self, num):
        if False:
            i = 10
            return i + 15
        num = [str(x) for x in num]
        num.sort(cmp=lambda x, y: cmp(y + x, x + y))
        largest = ''.join(num)
        return largest.lstrip('0') or '0'