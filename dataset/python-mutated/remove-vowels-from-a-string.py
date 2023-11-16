class Solution(object):

    def removeVowels(self, S):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type S: str\n        :rtype: str\n        '
        lookup = set('aeiou')
        return ''.join((c for c in S if c not in lookup))