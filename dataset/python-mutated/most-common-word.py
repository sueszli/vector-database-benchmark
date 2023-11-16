import collections

class Solution(object):

    def mostCommonWord(self, paragraph, banned):
        if False:
            while True:
                i = 10
        '\n        :type paragraph: str\n        :type banned: List[str]\n        :rtype: str\n        '
        lookup = set(banned)
        counts = collections.Counter((word.strip("!?',.") for word in paragraph.lower().split()))
        result = ''
        for word in counts:
            if (not result or counts[word] > counts[result]) and word not in lookup:
                result = word
        return result