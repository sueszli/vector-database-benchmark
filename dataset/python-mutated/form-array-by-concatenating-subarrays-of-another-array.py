class Solution(object):

    def canChoose(self, groups, nums):
        if False:
            print('Hello World!')
        '\n        :type groups: List[List[int]]\n        :type nums: List[int]\n        :rtype: bool\n        '

        def getPrefix(pattern):
            if False:
                while True:
                    i = 10
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j + 1 > 0 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix

        def KMP(text, pattern, start):
            if False:
                print('Hello World!')
            prefix = getPrefix(pattern)
            j = -1
            for i in xrange(start, len(text)):
                while j + 1 > 0 and pattern[j + 1] != text[i]:
                    j = prefix[j]
                if pattern[j + 1] == text[i]:
                    j += 1
                if j + 1 == len(pattern):
                    return i - j
            return -1
        pos = 0
        for group in groups:
            pos = KMP(nums, group, pos)
            if pos == -1:
                return False
            pos += len(group)
        return True