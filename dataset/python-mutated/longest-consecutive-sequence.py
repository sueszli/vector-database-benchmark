class Solution(object):

    def longestConsecutive(self, num):
        if False:
            print('Hello World!')
        (result, lengths) = (1, {key: 0 for key in num})
        for i in num:
            if lengths[i] == 0:
                lengths[i] = 1
                (left, right) = (lengths.get(i - 1, 0), lengths.get(i + 1, 0))
                length = 1 + left + right
                (result, lengths[i - left], lengths[i + right]) = (max(result, length), length, length)
        return result