class Solution(object):

    def getMaxRepetitions(self, s1, n1, s2, n2):
        if False:
            print('Hello World!')
        '\n        :type s1: str\n        :type n1: int\n        :type s2: str\n        :type n2: int\n        :rtype: int\n        '
        repeat_count = [0] * (len(s2) + 1)
        lookup = {}
        (j, count) = (0, 0)
        for k in xrange(1, n1 + 1):
            for i in xrange(len(s1)):
                if s1[i] == s2[j]:
                    j = (j + 1) % len(s2)
                    count += j == 0
            if j in lookup:
                i = lookup[j]
                prefix_count = repeat_count[i]
                pattern_count = (count - repeat_count[i]) * ((n1 - i) // (k - i))
                suffix_count = repeat_count[i + (n1 - i) % (k - i)] - repeat_count[i]
                return (prefix_count + pattern_count + suffix_count) / n2
            lookup[j] = k
            repeat_count[k] = count
        return repeat_count[n1] / n2