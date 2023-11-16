class Solution(object):

    def maxNumberOfAlloys(self, n, k, budget, composition, stock, cost):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :type budget: int\n        :type composition: List[List[int]]\n        :type stock: List[int]\n        :type cost: List[int]\n        :rtype: int\n        '

        def count(machine, budget):
            if False:
                while True:
                    i = 10

            def cnt(x):
                if False:
                    while True:
                        i = 10
                return stock[x] // machine[x]
            idxs = range(n)
            idxs.sort(key=cnt)
            result = cnt(idxs[0])
            prefix = curr = discount = 0
            for i in xrange(n):
                curr += cost[idxs[i]] * machine[idxs[i]]
                discount += cost[idxs[i]] * (stock[idxs[i]] % machine[idxs[i]])
                if i + 1 != n and cnt(idxs[i + 1]) - cnt(idxs[i]) == 0:
                    continue
                prefix += curr
                budget += discount
                curr = discount = 0
                mn = min(cnt(idxs[i + 1]) - cnt(idxs[i]) if i + 1 < n else float('inf'), budget // prefix)
                if mn == 0:
                    break
                budget -= prefix * mn
                result += mn
            return result
        return max((count(machine, budget) for machine in composition))

class Solution2(object):

    def maxNumberOfAlloys(self, n, k, budget, composition, stock, cost):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :type budget: int\n        :type composition: List[List[int]]\n        :type stock: List[int]\n        :type cost: List[int]\n        :rtype: int\n        '

        def check(x):
            if False:
                for i in range(10):
                    print('nop')
            for machine in composition:
                curr = 0
                for i in xrange(n):
                    curr += max(x * machine[i] - stock[i], 0) * cost[i]
                    if curr > budget:
                        break
                if curr <= budget:
                    return True
            return False
        (left, right) = (1, min(stock) + budget)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right