class Solution(object):

    def numberOfWeeks(self, milestones):
        if False:
            print('Hello World!')
        '\n        :type milestones: List[int]\n        :rtype: int\n        '
        (total, max_num) = (sum(milestones), max(milestones))
        other_total = total - max_num
        return other_total + min(other_total + 1, max_num)