class Solution(object):

    def haveConflict(self, event1, event2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type event1: List[str]\n        :type event2: List[str]\n        :rtype: bool\n        '
        return max(event1[0], event2[0]) <= min(event1[1], event2[1])