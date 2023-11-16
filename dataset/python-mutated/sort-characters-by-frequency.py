import collections

class Solution(object):

    def frequencySort(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: str\n        '
        freq = collections.defaultdict(int)
        for c in s:
            freq[c] += 1
        counts = [''] * (len(s) + 1)
        for c in freq:
            counts[freq[c]] += c
        result = ''
        for count in reversed(xrange(len(counts) - 1)):
            for c in counts[count]:
                result += c * count
        return result