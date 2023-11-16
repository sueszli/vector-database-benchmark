import collections

class Solution(object):

    def maxNumberOfBalloons(self, text):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type text: str\n        :rtype: int\n        '
        TARGET = 'balloon'
        source_count = collections.Counter(text)
        target_count = collections.Counter(TARGET)
        return min((source_count[c] // target_count[c] for c in target_count.iterkeys()))