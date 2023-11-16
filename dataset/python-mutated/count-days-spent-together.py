class Solution(object):

    def countDaysTogether(self, arriveAlice, leaveAlice, arriveBob, leaveBob):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arriveAlice: str\n        :type leaveAlice: str\n        :type arriveBob: str\n        :type leaveBob: str\n        :rtype: int\n        '
        NUMS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        prefix = [0] * (len(NUMS) + 1)
        for i in xrange(len(NUMS)):
            prefix[i + 1] += prefix[i] + NUMS[i]

        def day(date):
            if False:
                print('Hello World!')
            return prefix[int(date[:2]) - 1] + int(date[3:])
        return max(day(min(leaveAlice, leaveBob)) - day(max(arriveAlice, arriveBob)) + 1, 0)