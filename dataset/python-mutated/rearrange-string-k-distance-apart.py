import collections
import itertools

class Solution(object):

    def rearrangeString(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :rtype: str\n        '
        if not k:
            return s
        cnts = collections.Counter(s)
        bucket_cnt = max(cnts.itervalues())
        if not (bucket_cnt - 1) * k + sum((x == bucket_cnt for x in cnts.itervalues())) <= len(s):
            return ''
        result = [0] * len(s)
        i = (len(s) - 1) % k
        for c in itertools.chain((c for (c, v) in cnts.iteritems() if v == bucket_cnt), (c for (c, v) in cnts.iteritems() if v != bucket_cnt)):
            for _ in xrange(cnts[c]):
                result[i] = c
                i += k
                if i >= len(result):
                    i = (i - 1) % k
        return ''.join(result)
import collections
import itertools

class Solution2(object):

    def rearrangeString(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type str: str\n        :type k: int\n        :rtype: str\n        '
        if not k:
            return s
        cnts = collections.Counter(s)
        bucket_cnt = (len(s) + k - 1) // k
        if not (max(cnts.itervalues()) <= bucket_cnt and cnts.values().count(bucket_cnt) <= (len(s) - 1) % k + 1):
            return ''
        result = [0] * len(s)
        i = 0
        for c in itertools.chain((c for (c, v) in cnts.iteritems() if v == bucket_cnt), (c for (c, v) in cnts.iteritems() if v <= bucket_cnt - 2), (c for (c, v) in cnts.iteritems() if v == bucket_cnt - 1)):
            for _ in xrange(cnts[c]):
                result[i] = c
                i += k
                if i >= len(result):
                    i = i % k + 1
        return ''.join(result)
import collections
import itertools

class Solution3(object):

    def rearrangeString(self, s, k):
        if False:
            i = 10
            return i + 15
        '\n        :type str: str\n        :type k: int\n        :rtype: str\n        '
        cnts = collections.Counter(s)
        bucket_cnt = max(cnts.itervalues())
        buckets = [[] for _ in xrange(bucket_cnt)]
        i = 0
        for c in itertools.chain((c for (c, v) in cnts.iteritems() if v == bucket_cnt), (c for (c, v) in cnts.iteritems() if v == bucket_cnt - 1), (c for (c, v) in cnts.iteritems() if v <= bucket_cnt - 2)):
            for _ in xrange(cnts[c]):
                buckets[i].append(c)
                i = (i + 1) % max(cnts[c], bucket_cnt - 1)
        if any((len(buckets[i]) < k for i in xrange(len(buckets) - 1))):
            return ''
        return ''.join(map(lambda x: ''.join(x), buckets))
from collections import Counter
from heapq import heappush, heappop

class Solution4(object):

    def rearrangeString(self, s, k):
        if False:
            print('Hello World!')
        '\n        :type str: str\n        :type k: int\n        :rtype: str\n        '
        if k <= 1:
            return s
        cnts = Counter(s)
        heap = []
        for (c, cnt) in cnts.iteritems():
            heappush(heap, [-cnt, c])
        result = []
        while heap:
            used_cnt_chars = []
            for _ in xrange(min(k, len(s) - len(result))):
                if not heap:
                    return ''
                cnt_char = heappop(heap)
                result.append(cnt_char[1])
                cnt_char[0] += 1
                if cnt_char[0] < 0:
                    used_cnt_chars.append(cnt_char)
            for cnt_char in used_cnt_chars:
                heappush(heap, cnt_char)
        return ''.join(result)