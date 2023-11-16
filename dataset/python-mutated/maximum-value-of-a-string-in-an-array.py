class Solution(object):

    def maximumValue(self, strs):
        if False:
            i = 10
            return i + 15
        '\n        :type strs: List[str]\n        :rtype: int\n        '
        return max((int(s) if s.isdigit() else len(s) for s in strs))