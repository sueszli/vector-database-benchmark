class Solution(object):

    def lastSubstring(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: str\n        '
        (left, right, l) = (0, 1, 0)
        while right + l < len(s):
            if s[left + l] == s[right + l]:
                l += 1
                continue
            if s[left + l] > s[right + l]:
                right += l + 1
            else:
                left = max(right, left + l + 1)
                right = left + 1
            l = 0
        return s[left:]
import collections

class Solution2(object):

    def lastSubstring(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: str\n        '
        count = collections.defaultdict(list)
        for i in xrange(len(s)):
            count[s[i]].append(i)
        max_c = max(count.iterkeys())
        starts = {}
        for i in count[max_c]:
            starts[i] = i + 1
        while len(starts) - 1 > 0:
            lookup = set()
            next_count = collections.defaultdict(list)
            for (start, end) in starts.iteritems():
                if end == len(s):
                    lookup.add(start)
                    continue
                next_count[s[end]].append(start)
                if end in starts:
                    lookup.add(end)
            next_starts = {}
            max_c = max(next_count.iterkeys())
            for start in next_count[max_c]:
                if start not in lookup:
                    next_starts[start] = starts[start] + 1
            starts = next_starts
        return s[next(starts.iterkeys()):]