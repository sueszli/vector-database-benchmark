import itertools

class Solution(object):

    def destCity(self, paths):
        if False:
            print('Hello World!')
        '\n        :type paths: List[List[str]]\n        :rtype: str\n        '
        (A, B) = map(set, itertools.izip(*paths))
        return (B - A).pop()