class Solution(object):

    def checkIfExist(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: bool\n        '
        lookup = set()
        for x in arr:
            if 2 * x in lookup or (x % 2 == 0 and x // 2 in lookup):
                return True
            lookup.add(x)
        return False