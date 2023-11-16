class Solution(object):

    def buildArray(self, target, n):
        if False:
            print('Hello World!')
        '\n        :type target: List[int]\n        :type n: int\n        :rtype: List[str]\n        '
        (result, curr) = ([], 1)
        for t in target:
            result.extend(['Push', 'Pop'] * (t - curr))
            result.append('Push')
            curr = t + 1
        return result