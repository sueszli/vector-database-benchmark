import collections

class Solution(object):

    def groupStrings(self, strings):
        if False:
            i = 10
            return i + 15
        groups = collections.defaultdict(list)
        for s in strings:
            groups[self.hashStr(s)].append(s)
        result = []
        for (key, val) in groups.iteritems():
            result.append(sorted(val))
        return result

    def hashStr(self, s):
        if False:
            i = 10
            return i + 15
        base = ord(s[0])
        hashcode = ''
        for i in xrange(len(s)):
            if ord(s[i]) - base >= 0:
                hashcode += unichr(ord('a') + ord(s[i]) - base)
            else:
                hashcode += unichr(ord('a') + ord(s[i]) - base + 26)
        return hashcode