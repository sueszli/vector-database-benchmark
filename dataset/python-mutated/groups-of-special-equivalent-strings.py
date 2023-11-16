class Solution(object):

    def numSpecialEquivGroups(self, A):
        if False:
            return 10
        '\n        :type A: List[str]\n        :rtype: int\n        '

        def count(word):
            if False:
                while True:
                    i = 10
            result = [0] * 52
            for (i, letter) in enumerate(word):
                result[ord(letter) - ord('a') + 26 * (i % 2)] += 1
            return tuple(result)
        return len({count(word) for word in A})