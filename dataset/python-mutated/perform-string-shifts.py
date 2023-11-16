class Solution(object):

    def stringShift(self, s, shift):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type shift: List[List[int]]\n        :rtype: str\n        '
        left_shifts = 0
        for (direction, amount) in shift:
            if not direction:
                left_shifts += amount
            else:
                left_shifts -= amount
        left_shifts %= len(s)
        return s[left_shifts:] + s[:left_shifts]