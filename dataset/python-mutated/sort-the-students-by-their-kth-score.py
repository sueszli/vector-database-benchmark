class Solution(object):

    def sortTheStudents(self, score, k):
        if False:
            print('Hello World!')
        '\n        :type score: List[List[int]]\n        :type k: int\n        :rtype: List[List[int]]\n        '
        score.sort(key=lambda x: x[k], reverse=True)
        return score