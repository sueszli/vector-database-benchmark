class Solution(object):

    def garbageCollection(self, garbage, travel):
        if False:
            print('Hello World!')
        '\n        :type garbage: List[str]\n        :type travel: List[int]\n        :rtype: int\n        '
        result = 0
        lookup = {}
        for i in xrange(len(garbage)):
            for c in garbage[i]:
                lookup[c] = i
            if i + 1 < len(travel):
                travel[i + 1] += travel[i]
            result += len(garbage[i])
        result += sum((travel[v - 1] for (_, v) in lookup.iteritems() if v - 1 >= 0))
        return result

class Solution2(object):

    def garbageCollection(self, garbage, travel):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type garbage: List[str]\n        :type travel: List[int]\n        :rtype: int\n        '
        result = 0
        for t in 'MPG':
            curr = 0
            for i in xrange(len(garbage)):
                cnt = garbage[i].count(t)
                if cnt:
                    result += curr + cnt
                    curr = 0
                if i < len(travel):
                    curr += travel[i]
        return result