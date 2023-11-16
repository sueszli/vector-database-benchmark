class Solution(object):

    def mostWordsFound(self, sentences):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type sentences: List[str]\n        :rtype: int\n        '
        return 1 + max((s.count(' ') for s in sentences))