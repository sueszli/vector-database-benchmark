import bisect

class Solution(object):

    def nextGreatestLetter(self, letters, target):
        if False:
            print('Hello World!')
        '\n        :type letters: List[str]\n        :type target: str\n        :rtype: str\n        '
        i = bisect.bisect_right(letters, target)
        return letters[0] if i == len(letters) else letters[i]