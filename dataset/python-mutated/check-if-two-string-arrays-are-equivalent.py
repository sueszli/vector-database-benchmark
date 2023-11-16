class Solution(object):

    def arrayStringsAreEqual(self, word1, word2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word1: List[str]\n        :type word2: List[str]\n        :rtype: bool\n        '
        idx1 = idx2 = arr_idx1 = arr_idx2 = 0
        while arr_idx1 < len(word1) and arr_idx2 < len(word2):
            if word1[arr_idx1][idx1] != word2[arr_idx2][idx2]:
                break
            idx1 += 1
            if idx1 == len(word1[arr_idx1]):
                idx1 = 0
                arr_idx1 += 1
            idx2 += 1
            if idx2 == len(word2[arr_idx2]):
                idx2 = 0
                arr_idx2 += 1
        return arr_idx1 == len(word1) and arr_idx2 == len(word2)