import bisect

class Solution(object):

    def minWastedSpace(self, packages, boxes):
        if False:
            print('Hello World!')
        '\n        :type packages: List[int]\n        :type boxes: List[List[int]]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        INF = float('inf')
        packages.sort()
        result = INF
        for box in boxes:
            box.sort()
            if box[-1] < packages[-1]:
                continue
            curr = left = 0
            for b in box:
                right = bisect.bisect_right(packages, b, left)
                curr += b * (right - left)
                left = right
            result = min(result, curr)
        return (result - sum(packages)) % MOD if result != INF else -1