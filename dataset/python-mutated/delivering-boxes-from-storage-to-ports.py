class Solution(object):

    def boxDelivering(self, boxes, portsCount, maxBoxes, maxWeight):
        if False:
            print('Hello World!')
        '\n        :type boxes: List[List[int]]\n        :type portsCount: int\n        :type maxBoxes: int\n        :type maxWeight: int\n        :rtype: int\n        '
        dp = [0] * (len(boxes) + 1)
        (left, cost, curr) = (0, 1, 0)
        for right in xrange(len(boxes)):
            if right == 0 or boxes[right][0] != boxes[right - 1][0]:
                cost += 1
            curr += boxes[right][1]
            while right - left + 1 > maxBoxes or curr > maxWeight or (left + 1 < right + 1 and dp[left + 1] == dp[left]):
                curr -= boxes[left][1]
                if boxes[left + 1][0] != boxes[left][0]:
                    cost -= 1
                left += 1
            dp[right + 1] = dp[left - 1 + 1] + cost
        return dp[len(boxes)]