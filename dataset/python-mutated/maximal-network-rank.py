class Solution(object):

    def maximalNetworkRank(self, n, roads):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :rtype: int\n        '
        MAX_N = 100
        MAX_NUM = MAX_N - 1

        def counting_sort(arr, key=lambda x: x, reverse=False):
            if False:
                while True:
                    i = 10
            count = [0] * (MAX_NUM + 1)
            for x in arr:
                count[key(x)] += 1
            for i in xrange(1, len(count)):
                count[i] += count[i - 1]
            result = [0] * len(arr)
            if not reverse:
                for x in reversed(arr):
                    count[key(x)] -= 1
                    result[count[key(x)]] = x
            else:
                for x in arr:
                    count[key(x)] -= 1
                    result[count[key(x)]] = x
                result.reverse()
            return result
        degree = [0] * n
        adj = collections.defaultdict(set)
        for (a, b) in roads:
            degree[a] += 1
            degree[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        sorted_idx = counting_sort(xrange(n), key=lambda x: degree[x], reverse=True)
        m = 2
        while m < n:
            if degree[sorted_idx[m]] != degree[sorted_idx[1]]:
                break
            m += 1
        result = degree[sorted_idx[0]] + degree[sorted_idx[1]] - 1
        for i in xrange(m - 1):
            for j in xrange(i + 1, m):
                if degree[sorted_idx[i]] + degree[sorted_idx[j]] - int(sorted_idx[i] in adj and sorted_idx[j] in adj[sorted_idx[i]]) > result:
                    return degree[sorted_idx[i]] + degree[sorted_idx[j]] - int(sorted_idx[i] in adj and sorted_idx[j] in adj[sorted_idx[i]])
        return result
import collections

class Solution2(object):

    def maximalNetworkRank(self, n, roads):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :rtype: int\n        '
        degree = [0] * n
        adj = collections.defaultdict(set)
        for (a, b) in roads:
            degree[a] += 1
            degree[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        sorted_idx = range(n)
        sorted_idx.sort(key=lambda x: -degree[x])
        m = 2
        while m < n:
            if degree[sorted_idx[m]] != degree[sorted_idx[1]]:
                break
            m += 1
        result = degree[sorted_idx[0]] + degree[sorted_idx[1]] - 1
        for i in xrange(m - 1):
            for j in xrange(i + 1, m):
                if degree[sorted_idx[i]] + degree[sorted_idx[j]] - int(sorted_idx[i] in adj and sorted_idx[j] in adj[sorted_idx[i]]) > result:
                    return degree[sorted_idx[i]] + degree[sorted_idx[j]] - int(sorted_idx[i] in adj and sorted_idx[j] in adj[sorted_idx[i]])
        return result
import collections

class Solution3(object):

    def maximalNetworkRank(self, n, roads):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :rtype: int\n        '
        degree = [0] * n
        adj = collections.defaultdict(set)
        for (a, b) in roads:
            degree[a] += 1
            degree[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        result = 0
        for i in xrange(n - 1):
            for j in xrange(i + 1, n):
                result = max(result, degree[i] + degree[j] - int(i in adj and j in adj[i]))
        return result