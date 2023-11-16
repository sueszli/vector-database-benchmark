class Solution(object):

    def findAnagrams(self, s, p):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type p: str\n        :rtype: List[int]\n        '
        result = []
        cnts = [0] * 26
        for c in p:
            cnts[ord(c) - ord('a')] += 1
        (left, right) = (0, 0)
        while right < len(s):
            cnts[ord(s[right]) - ord('a')] -= 1
            while left <= right and cnts[ord(s[right]) - ord('a')] < 0:
                cnts[ord(s[left]) - ord('a')] += 1
                left += 1
            if right - left + 1 == len(p):
                result.append(left)
            right += 1
        return result