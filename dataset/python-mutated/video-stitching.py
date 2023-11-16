class Solution(object):

    def videoStitching(self, clips, T):
        if False:
            return 10
        '\n        :type clips: List[List[int]]\n        :type T: int\n        :rtype: int\n        '
        if T == 0:
            return 0
        result = 1
        (curr_reachable, reachable) = (0, 0)
        clips.sort()
        for (left, right) in clips:
            if left > reachable:
                break
            elif left > curr_reachable:
                curr_reachable = reachable
                result += 1
            reachable = max(reachable, right)
            if reachable >= T:
                return result
        return -1