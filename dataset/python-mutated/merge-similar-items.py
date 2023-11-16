class Solution(object):

    def mergeSimilarItems(self, items1, items2):
        if False:
            return 10
        '\n        :type items1: List[List[int]]\n        :type items2: List[List[int]]\n        :rtype: List[List[int]]\n        '
        return sorted((Counter(dict(items1)) + Counter(dict(items2))).iteritems())