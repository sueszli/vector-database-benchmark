class Solution(object):

    def lengthOfLongestSubstringKDistinct(self, s, k):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        (longest, start, distinct_count, visited) = (0, 0, 0, [0 for _ in xrange(256)])
        for (i, char) in enumerate(s):
            if visited[ord(char)] == 0:
                distinct_count += 1
            visited[ord(char)] += 1
            while distinct_count > k:
                visited[ord(s[start])] -= 1
                if visited[ord(s[start])] == 0:
                    distinct_count -= 1
                start += 1
            longest = max(longest, i - start + 1)
        return longest
from collections import Counter

class Solution2(object):

    def lengthOfLongestSubstringKDistinct(self, s, k):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        counter = Counter()
        (left, max_length) = (0, 0)
        for (right, char) in enumerate(s):
            counter[char] += 1
            while len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
            max_length = max(max_length, right - left + 1)
        return max_length