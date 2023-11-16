class Solution(object):

    def countSeniors(self, details):
        if False:
            return 10
        '\n        :type details: List[str]\n        :rtype: int\n        '
        return sum((x[-4:-2] > '60' for x in details))