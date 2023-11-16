import collections
import itertools

class Solution(object):

    def largestWordCount(self, messages, senders):
        if False:
            i = 10
            return i + 15
        '\n        :type messages: List[str]\n        :type senders: List[str]\n        :rtype: str\n        '
        cnt = collections.Counter()
        for (m, s) in itertools.izip(messages, senders):
            cnt[s] += m.count(' ') + 1
        return max((k for k in cnt.iterkeys()), key=lambda x: (cnt[x], x))