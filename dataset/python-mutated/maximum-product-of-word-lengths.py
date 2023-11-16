class Solution(object):

    def maxProduct(self, words):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :rtype: int\n        '

        def counting_sort(words):
            if False:
                return 10
            k = 1000
            buckets = [[] for _ in xrange(k)]
            for word in words:
                buckets[len(word)].append(word)
            res = []
            for i in reversed(xrange(k)):
                if buckets[i]:
                    res += buckets[i]
            return res
        words = counting_sort(words)
        bits = [0] * len(words)
        for (i, word) in enumerate(words):
            for c in word:
                bits[i] |= 1 << ord(c) - ord('a')
        max_product = 0
        for i in xrange(len(words) - 1):
            if len(words[i]) ** 2 <= max_product:
                break
            for j in xrange(i + 1, len(words)):
                if len(words[i]) * len(words[j]) <= max_product:
                    break
                if not bits[i] & bits[j]:
                    max_product = len(words[i]) * len(words[j])
        return max_product

class Solution2(object):

    def maxProduct(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :rtype: int\n        '
        words.sort(key=lambda x: len(x), reverse=True)
        bits = [0] * len(words)
        for (i, word) in enumerate(words):
            for c in word:
                bits[i] |= 1 << ord(c) - ord('a')
        max_product = 0
        for i in xrange(len(words) - 1):
            if len(words[i]) ** 2 <= max_product:
                break
            for j in xrange(i + 1, len(words)):
                if len(words[i]) * len(words[j]) <= max_product:
                    break
                if not bits[i] & bits[j]:
                    max_product = len(words[i]) * len(words[j])
        return max_product