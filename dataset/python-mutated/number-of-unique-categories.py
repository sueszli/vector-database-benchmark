class CategoryHandler:

    def haveSameCategory(self, a, b):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def numberOfCategories(self, n, categoryHandler):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type categoryHandler: CategoryHandler\n        :rtype: int\n        '
        return sum((all((not categoryHandler.haveSameCategory(j, i) for j in xrange(i))) for i in xrange(n)))