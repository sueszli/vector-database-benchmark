import collections

class Solution(object):

    def maxConsecutiveAnswers(self, answerKey, k):
        if False:
            while True:
                i = 10
        '\n        :type answerKey: str\n        :type k: int\n        :rtype: int\n        '
        result = max_count = 0
        count = collections.Counter()
        for i in xrange(len(answerKey)):
            count[answerKey[i]] += 1
            max_count = max(max_count, count[answerKey[i]])
            if result - max_count >= k:
                count[answerKey[i - result]] -= 1
            else:
                result += 1
        return result