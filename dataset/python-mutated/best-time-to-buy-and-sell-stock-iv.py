import random

class Solution(object):

    def maxProfit(self, k, prices):
        if False:
            print('Hello World!')
        '\n        :type k: int\n        :type prices: List[int]\n        :rtype: int\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                while True:
                    i = 10

            def tri_partition(nums, left, right, target, compare):
                if False:
                    return 10
                mid = left
                while mid <= right:
                    if nums[mid] == target:
                        mid += 1
                    elif compare(nums[mid], target):
                        (nums[left], nums[mid]) = (nums[mid], nums[left])
                        left += 1
                        mid += 1
                    else:
                        (nums[mid], nums[right]) = (nums[right], nums[mid])
                        right -= 1
                return (left, right)
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                pivot_idx = random.randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1
        profits = []
        v_p_stk = []
        (v, p) = (-1, -1)
        while p + 1 < len(prices):
            for v in xrange(p + 1, len(prices) - 1):
                if prices[v] < prices[v + 1]:
                    break
            else:
                v = len(prices) - 1
            for p in xrange(v, len(prices) - 1):
                if prices[p] > prices[p + 1]:
                    break
            else:
                p = len(prices) - 1
            while v_p_stk and prices[v_p_stk[-1][0]] > prices[v]:
                (last_v, last_p) = v_p_stk.pop()
                profits.append(prices[last_p] - prices[last_v])
            while v_p_stk and prices[v_p_stk[-1][1]] <= prices[p]:
                (last_v, last_p) = v_p_stk.pop()
                profits.append(prices[last_p] - prices[v])
                v = last_v
            v_p_stk.append((v, p))
        while v_p_stk:
            (last_v, last_p) = v_p_stk.pop()
            profits.append(prices[last_p] - prices[last_v])
        if k > len(profits):
            k = len(profits)
        else:
            nth_element(profits, k - 1, compare=lambda a, b: a > b)
        return sum((profits[i] for i in xrange(k)))

class Solution2(object):

    def maxProfit(self, k, prices):
        if False:
            while True:
                i = 10
        '\n        :type k: int\n        :type prices: List[int]\n        :rtype: int\n        '

        def maxAtMostNPairsProfit(sprices):
            if False:
                return 10
            profit = 0
            for i in xrange(len(prices) - 1):
                profit += max(0, prices[i + 1] - prices[i])
            return profit

        def maxAtMostKPairsProfit(prices, k):
            if False:
                return 10
            max_buy = [float('-inf') for _ in xrange(k + 1)]
            max_sell = [0 for _ in xrange(k + 1)]
            for i in xrange(len(prices)):
                for j in xrange(1, k + 1):
                    max_buy[j] = max(max_buy[j], max_sell[j - 1] - prices[i])
                    max_sell[j] = max(max_sell[j], max_buy[j] + prices[i])
            return max_sell[k]
        if k >= len(prices) // 2:
            return maxAtMostNPairsProfit(prices)
        return maxAtMostKPairsProfit(prices, k)