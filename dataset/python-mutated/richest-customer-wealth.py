import itertools

class Solution(object):

    def maximumWealth(self, accounts):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type accounts: List[List[int]]\n        :rtype: int\n        '
        return max(itertools.imap(sum, accounts))