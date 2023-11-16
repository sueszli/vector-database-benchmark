# Time:  O(n^3) ~ O(n^4)
# Space: O(n^3)

class Solution(object):
    def removeBoxes(self, boxes):
        """
        :type boxes: List[int]
        :rtype: int
        """
        def dfs(boxes, l, r, k, lookup):
            if l > r: return 0
            if lookup[l][r][k]: return lookup[l][r][k]

            ll, kk = l, k
            while l < r and boxes[l+1] == boxes[l]:
                l += 1
                k += 1
            result = dfs(boxes, l+1, r, 0, lookup) + (k+1) ** 2
            for i in xrange(l+1, r+1):
                if boxes[i] == boxes[l]:
                    result = max(result, dfs(boxes, l+1, i-1, 0, lookup) + dfs(boxes, i, r, k+1, lookup))
            lookup[ll][r][kk] = result
            return result

        lookup = [[[0]*len(boxes) for _ in xrange(len(boxes)) ] for _ in xrange(len(boxes)) ]
        return dfs(boxes, 0, len(boxes)-1, 0, lookup)

