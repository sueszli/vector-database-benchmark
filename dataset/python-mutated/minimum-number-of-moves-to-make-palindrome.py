class BIT(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.__bit = [0] * (n + 1)

    def add(self, i, val):
        if False:
            i = 10
            return i + 15
        i += 1
        while i < len(self.__bit):
            self.__bit[i] += val
            i += i & -i

    def query(self, i):
        if False:
            for i in range(10):
                print('nop')
        i += 1
        ret = 0
        while i > 0:
            ret += self.__bit[i]
            i -= i & -i
        return ret

class Solution(object):

    def minMovesToMakePalindrome(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        idxs = [[] for _ in xrange(26)]
        for (i, c) in enumerate(s):
            idxs[ord(c) - ord('a')].append(i)
        (targets, pairs) = ([0] * len(s), [])
        for (c, idx) in enumerate(idxs):
            for i in xrange(len(idx) // 2):
                pairs.append((idx[i], idx[~i]))
            if len(idx) % 2:
                targets[idx[len(idx) // 2]] = len(s) // 2
        pairs.sort()
        for (i, (l, r)) in enumerate(pairs):
            (targets[l], targets[r]) = (i, len(s) - 1 - i)
        bit = BIT(len(s))
        result = 0
        for i in targets:
            result += i - bit.query(i - 1)
            bit.add(i, 1)
        return result

class Solution2(object):

    def minMovesToMakePalindrome(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        s = list(s)
        result = 0
        while s:
            i = s.index(s[-1])
            if i == len(s) - 1:
                result += i // 2
            else:
                result += i
                s.pop(i)
            s.pop()
        return result