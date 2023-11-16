class Solution(object):

    def maximumRows(self, matrix, numSelect):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type matrix: List[List[int]]\n        :type numSelect: int\n        :rtype: int\n        '

        def next_popcount(n):
            if False:
                for i in range(10):
                    print('nop')
            lowest_bit = n & -n
            left_bits = n + lowest_bit
            changed_bits = n ^ left_bits
            right_bits = changed_bits // lowest_bit >> 2
            return left_bits | right_bits
        masks = [reduce(lambda m, c: m | matrix[r][-1 - c] << c, xrange(len(matrix[0])), 0) for r in xrange(len(matrix))]
        result = 0
        mask = (1 << numSelect) - 1
        while mask < 1 << len(matrix[0]):
            result = max(result, sum((m & mask == m for m in masks)))
            mask = next_popcount(mask)
        return result