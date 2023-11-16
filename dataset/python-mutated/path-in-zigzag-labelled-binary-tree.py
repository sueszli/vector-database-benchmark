class Solution(object):

    def pathInZigZagTree(self, label):
        if False:
            return 10
        '\n        :type label: int\n        :rtype: List[int]\n        '
        count = 2 ** label.bit_length()
        result = []
        while label >= 1:
            result.append(label)
            label = (count // 2 + (count - 1 - label)) // 2
            count //= 2
        result.reverse()
        return result