import collections

class Solution(object):

    def findSubstring(self, s, words):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type words: List[str]\n        :rtype: List[int]\n        '
        if not words:
            return []
        (result, m, n, k) = ([], len(s), len(words), len(words[0]))
        if m < n * k:
            return result
        lookup = collections.defaultdict(int)
        for i in words:
            lookup[i] += 1
        for i in xrange(k):
            (left, count) = (i, 0)
            tmp = collections.defaultdict(int)
            for j in xrange(i, m - k + 1, k):
                s1 = s[j:j + k]
                if s1 in lookup:
                    tmp[s1] += 1
                    count += 1
                    while tmp[s1] > lookup[s1]:
                        tmp[s[left:left + k]] -= 1
                        count -= 1
                        left += k
                    if count == n:
                        result.append(left)
                else:
                    tmp = collections.defaultdict(int)
                    count = 0
                    left = j + k
        return result

class Solution2(object):

    def findSubstring(self, s, words):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type words: List[str]\n        :rtype: List[int]\n        '
        (result, m, n, k) = ([], len(s), len(words), len(words[0]))
        if m < n * k:
            return result
        lookup = collections.defaultdict(int)
        for i in words:
            lookup[i] += 1
        for i in xrange(m + 1 - k * n):
            (cur, j) = (collections.defaultdict(int), 0)
            while j < n:
                word = s[i + j * k:i + j * k + k]
                if word not in lookup:
                    break
                cur[word] += 1
                if cur[word] > lookup[word]:
                    break
                j += 1
            if j == n:
                result.append(i)
        return result