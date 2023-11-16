class Solution(object):

    def minAddToMakeValid(self, S):
        if False:
            return 10
        '\n        :type S: str\n        :rtype: int\n        '
        (add, bal) = (0, 0)
        for c in S:
            bal += 1 if c == '(' else -1
            if bal == -1:
                add += 1
                bal += 1
        return add + bal