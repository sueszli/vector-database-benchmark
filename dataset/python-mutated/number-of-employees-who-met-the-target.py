class Solution(object):

    def numberOfEmployeesWhoMetTarget(self, hours, target):
        if False:
            print('Hello World!')
        '\n        :type hours: List[int]\n        :type target: int\n        :rtype: int\n        '
        return sum((x >= target for x in hours))