class Solution(object):

    def maxProfit(self, inventory, orders):
        if False:
            print('Hello World!')
        '\n        :type inventory: List[int]\n        :type orders: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def check(inventory, orders, x):
            if False:
                return 10
            return count(inventory, x) > orders

        def count(inventory, x):
            if False:
                print('Hello World!')
            return sum((count - x + 1 for count in inventory if count >= x))
        (left, right) = (1, max(inventory))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(inventory, orders, mid):
                right = mid - 1
            else:
                left = mid + 1
        return (sum(((left + cnt) * (cnt - left + 1) // 2 for cnt in inventory if cnt >= left)) + (left - 1) * (orders - count(inventory, left))) % MOD