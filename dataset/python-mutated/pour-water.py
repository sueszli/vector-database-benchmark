class Solution(object):

    def pourWater(self, heights, V, K):
        if False:
            print('Hello World!')
        '\n        :type heights: List[int]\n        :type V: int\n        :type K: int\n        :rtype: List[int]\n        '
        for _ in xrange(V):
            best = K
            for d in (-1, 1):
                i = K
                while 0 <= i + d < len(heights) and heights[i + d] <= heights[i]:
                    if heights[i + d] < heights[i]:
                        best = i + d
                    i += d
                if best != K:
                    break
            heights[best] += 1
        return heights