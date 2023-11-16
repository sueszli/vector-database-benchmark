class Solution(object):

    def maximumCoins(self, heroes, monsters, coins):
        if False:
            i = 10
            return i + 15
        '\n        :type heroes: List[int]\n        :type monsters: List[int]\n        :type coins: List[int]\n        :rtype: List[int]\n        '
        idxs1 = range(len(heroes))
        idxs1.sort(key=lambda x: heroes[x])
        idxs2 = range(len(monsters))
        idxs2.sort(key=lambda x: monsters[x])
        result = [0] * len(idxs1)
        i = curr = 0
        for idx in idxs1:
            for i in xrange(i, len(idxs2)):
                if monsters[idxs2[i]] > heroes[idx]:
                    break
                curr += coins[idxs2[i]]
            else:
                i = len(idxs2)
            result[idx] = curr
        return result