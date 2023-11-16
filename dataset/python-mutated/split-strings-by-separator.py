class Solution(object):

    def splitWordsBySeparator(self, words, separator):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :type separator: str\n        :rtype: List[str]\n        '
        return [w for word in words for w in word.split(separator) if w]