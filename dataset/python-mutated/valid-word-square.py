class Solution(object):

    def validWordSquare(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :rtype: bool\n        '
        for i in xrange(len(words)):
            for j in xrange(len(words[i])):
                if j >= len(words) or i >= len(words[j]) or words[j][i] != words[i][j]:
                    return False
        return True