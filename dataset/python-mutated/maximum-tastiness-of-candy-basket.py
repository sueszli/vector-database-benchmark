class Solution(object):

    def maximumTastiness(self, price, k):
        if False:
            print('Hello World!')
        '\n        :type price: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(x):
            if False:
                for i in range(10):
                    print('nop')
            cnt = prev = 0
            for i in xrange(len(price)):
                if prev and price[i] - prev < x:
                    continue
                cnt += 1
                if cnt == k:
                    break
                prev = price[i]
            return cnt >= k
        price.sort()
        (left, right) = (1, price[-1] - price[0])
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right