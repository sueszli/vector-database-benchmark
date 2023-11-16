class Solution(object):

    def minimumCost(self, sentence, k):
        if False:
            i = 10
            return i + 15
        '\n        :type sentence: str\n        :type k: int\n        :rtype: int\n        '

        def lens(sentence):
            if False:
                print('Hello World!')
            j = len(sentence) - 1
            for i in reversed(xrange(-1, len(sentence))):
                if i == -1 or sentence[i] == ' ':
                    yield (j - i)
                    j = i - 1
        (word_lens, dp) = ([], [])
        t = -1
        for l in lens(sentence):
            word_lens.append(l)
            dp.append(float('inf'))
            t += l + 1
            if t <= k:
                dp[-1] = 0
                continue
            total = l
            for j in reversed(xrange(len(dp) - 1)):
                dp[-1] = min(dp[-1], dp[j] + (k - total) ** 2)
                total += word_lens[j] + 1
                if total > k:
                    word_lens = word_lens[j:]
                    dp = dp[j:]
                    break
        return dp[-1] if dp else 0

class Solution2(object):

    def minimumCost(self, sentence, k):
        if False:
            while True:
                i = 10
        '\n        :type sentence: str\n        :type k: int\n        :rtype: int\n        '
        word_lens = []
        j = 0
        for i in xrange(len(sentence) + 1):
            if i != len(sentence) and sentence[i] != ' ':
                continue
            word_lens.append(i - j)
            j = i + 1
        dp = [float('inf')] * len(word_lens)
        (i, total) = (len(word_lens) - 1, -1)
        while i >= 0 and total + (word_lens[i] + 1) <= k:
            total += word_lens[i] + 1
            dp[i] = 0
            i -= 1
        for i in reversed(xrange(i + 1)):
            total = word_lens[i]
            for j in xrange(i + 1, len(dp)):
                dp[i] = min(dp[i], dp[j] + (k - total) ** 2)
                total += word_lens[j] + 1
                if total > k:
                    break
        return dp[0]

class Solution3(object):

    def minimumCost(self, sentence, k):
        if False:
            i = 10
            return i + 15
        '\n        :type sentence: str\n        :type k: int\n        :rtype: int\n        '
        word_lens = []
        j = 0
        for i in xrange(len(sentence) + 1):
            if i != len(sentence) and sentence[i] != ' ':
                continue
            word_lens.append(i - j)
            j = i + 1
        dp = [float('inf')] * (1 + (len(word_lens) - 1))
        dp[0] = 0
        for i in xrange(1, len(word_lens) - 1 + 1):
            total = word_lens[i - 1]
            for j in reversed(xrange(i)):
                dp[i] = min(dp[i], dp[j] + (k - total) ** 2)
                if j - 1 < 0:
                    continue
                total += word_lens[j - 1] + 1
                if total > k:
                    break
        (i, total) = (len(word_lens) - 1, -1)
        while i >= 0 and total + (word_lens[i] + 1) <= k:
            total += word_lens[i] + 1
            i -= 1
        return min((dp[j] for j in xrange(i + 1, len(dp))))