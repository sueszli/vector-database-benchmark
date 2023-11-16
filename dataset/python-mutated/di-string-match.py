class Solution(object):

    def diStringMatch(self, S):
        if False:
            i = 10
            return i + 15
        '\n        :type S: str\n        :rtype: List[int]\n        '
        result = []
        (left, right) = (0, len(S))
        for c in S:
            if c == 'I':
                result.append(left)
                left += 1
            else:
                result.append(right)
                right -= 1
        result.append(left)
        return result