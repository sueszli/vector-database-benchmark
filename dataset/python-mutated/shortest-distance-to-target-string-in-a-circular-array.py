class Solution(object):

    def closetTarget(self, words, target, startIndex):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :type target: str\n        :type startIndex: int\n        :rtype: int\n        '
        INF = float('inf')
        result = INF
        for (i, w) in enumerate(words):
            if w == target:
                result = min(result, (i - startIndex) % len(words), (startIndex - i) % len(words))
        return result if result != INF else -1