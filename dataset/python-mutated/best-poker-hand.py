class Solution(object):

    def bestHand(self, ranks, suits):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type ranks: List[int]\n        :type suits: List[str]\n        :rtype: str\n        '
        LOOKUP = ['', 'High Card', 'Pair', 'Three of a Kind', 'Three of a Kind', 'Three of a Kind']
        if all((suits[i] == suits[0] for i in xrange(1, len(suits)))):
            return 'Flush'
        cnt = [0] * 13
        for x in ranks:
            cnt[x - 1] += 1
        return LOOKUP[max(cnt)]