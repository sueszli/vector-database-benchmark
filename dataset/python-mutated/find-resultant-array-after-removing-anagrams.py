import collections

class Solution(object):

    def removeAnagrams(self, words):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        result = []
        prev = None
        for x in words:
            cnt = collections.Counter(x)
            if prev and prev == cnt:
                continue
            prev = cnt
            result.append(x)
        return result
import collections

class Solution2(object):

    def removeAnagrams(self, words):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        result = []
        prev = None
        for x in words:
            s = sorted(x)
            if prev and prev == s:
                continue
            prev = s
            result.append(x)
        return result
import collections

class Solution3(object):

    def removeAnagrams(self, words):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :rtype: List[str]\n        '
        return [words[i] for i in xrange(len(words)) if i == 0 or sorted(words[i - 1]) != sorted(words[i])]