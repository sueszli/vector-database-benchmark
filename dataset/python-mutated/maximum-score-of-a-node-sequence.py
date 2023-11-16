import heapq

class Solution(object):

    def maximumScore(self, scores, edges):
        if False:
            print('Hello World!')
        '\n        :type scores: List[int]\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def find_top3(scores, x, top3):
            if False:
                for i in range(10):
                    print('nop')
            heapq.heappush(top3, (scores[x], x))
            if len(top3) > 3:
                heapq.heappop(top3)
        top3 = [[] for _ in xrange(len(scores))]
        for (a, b) in edges:
            find_top3(scores, b, top3[a])
            find_top3(scores, a, top3[b])
        result = -1
        for (a, b) in edges:
            for (_, c) in top3[a]:
                if c == b:
                    continue
                for (_, d) in top3[b]:
                    if d == a or d == c:
                        continue
                    result = max(result, sum((scores[x] for x in (a, b, c, d))))
        return result