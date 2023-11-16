import collections

class Solution(object):

    def dividePlayers(self, skill):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type skill: List[int]\n        :rtype: int\n        '
        target = sum(skill) // (len(skill) // 2)
        cnt = collections.Counter(skill)
        result = 0
        for (k, v) in cnt.iteritems():
            if target - k not in cnt or cnt[target - k] != cnt[k]:
                return -1
            result += k * (target - k) * v
        return result // 2