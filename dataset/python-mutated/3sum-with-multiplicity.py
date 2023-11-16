import collections
import itertools

class Solution(object):

    def threeSumMulti(self, A, target):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :type target: int\n        :rtype: int\n        '
        count = collections.Counter(A)
        result = 0
        for (i, j) in itertools.combinations_with_replacement(count, 2):
            k = target - i - j
            if i == j == k:
                result += count[i] * (count[i] - 1) * (count[i] - 2) // 6
            elif i == j != k:
                result += count[i] * (count[i] - 1) // 2 * count[k]
            elif max(i, j) < k:
                result += count[i] * count[j] * count[k]
        return result % (10 ** 9 + 7)