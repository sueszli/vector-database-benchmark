import itertools

class Solution(object):

    def getWordsInLongestSubsequence(self, n, words, groups):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type words: List[str]\n        :type groups: List[int]\n        :rtype: List[str]\n        '

        def check(s1, s2):
            if False:
                while True:
                    i = 10
            return len(s1) == len(s2) and sum((a != b for (a, b) in itertools.izip(s1, s2))) == 1
        dp = [[1, -1] for _ in xrange(n)]
        for i in reversed(xrange(n)):
            for j in xrange(i + 1, n):
                if groups[i] != groups[j] and check(words[j], words[i]):
                    dp[i] = max(dp[i], [dp[j][0] + 1, j])
        result = []
        i = max(xrange(n), key=lambda x: dp[x])
        while i != -1:
            result.append(words[i])
            i = dp[i][1]
        return result
import itertools

class Solution2(object):

    def getWordsInLongestSubsequence(self, n, words, groups):
        if False:
            return 10
        '\n        :type n: int\n        :type words: List[str]\n        :type groups: List[int]\n        :rtype: List[str]\n        '

        def check(s1, s2):
            if False:
                for i in range(10):
                    print('nop')
            return len(s1) == len(s2) and sum((a != b for (a, b) in itertools.izip(s1, s2))) == 1
        dp = [[1, -1] for _ in xrange(n)]
        for i in xrange(n):
            for j in xrange(i):
                if groups[i] != groups[j] and check(words[j], words[i]):
                    dp[i] = max(dp[i], [dp[j][0] + 1, j])
        result = []
        i = max(xrange(n), key=lambda x: dp[x])
        while i != -1:
            result.append(words[i])
            i = dp[i][1]
        result.reverse()
        return result
import itertools

class Solution3(object):

    def getWordsInLongestSubsequence(self, n, words, groups):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type words: List[str]\n        :type groups: List[int]\n        :rtype: List[str]\n        '

        def check(s1, s2):
            if False:
                for i in range(10):
                    print('nop')
            return len(s1) == len(s2) and sum((a != b for (a, b) in itertools.izip(s1, s2))) == 1
        dp = [[] for _ in xrange(n)]
        for i in xrange(n):
            for j in xrange(i):
                if groups[i] != groups[j] and check(words[j], words[i]) and (len(dp[j]) > len(dp[i])):
                    dp[i] = dp[j]
            dp[i] = dp[i] + [i]
        return map(lambda x: words[x], max(dp, key=lambda x: len(x)))