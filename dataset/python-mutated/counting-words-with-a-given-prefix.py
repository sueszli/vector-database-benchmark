class Solution(object):

    def prefixCount(self, words, pref):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :type pref: str\n        :rtype: int\n        '
        return sum((x.startswith(pref) for x in words))