import collections

class Solution(object):

    def deckRevealedIncreasing(self, deck):
        if False:
            print('Hello World!')
        '\n        :type deck: List[int]\n        :rtype: List[int]\n        '
        d = collections.deque()
        deck.sort(reverse=True)
        for i in deck:
            if d:
                d.appendleft(d.pop())
            d.appendleft(i)
        return list(d)