class Node(object):

    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        if False:
            return 10
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight

class Solution(object):

    def construct(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: Node\n        '

        def dfs(grid, x, y, l):
            if False:
                for i in range(10):
                    print('nop')
            if l == 1:
                return Node(grid[x][y] == 1, True, None, None, None, None)
            half = l // 2
            topLeftNode = dfs(grid, x, y, half)
            topRightNode = dfs(grid, x, y + half, half)
            bottomLeftNode = dfs(grid, x + half, y, half)
            bottomRightNode = dfs(grid, x + half, y + half, half)
            if topLeftNode.isLeaf and topRightNode.isLeaf and bottomLeftNode.isLeaf and bottomRightNode.isLeaf and (topLeftNode.val == topRightNode.val == bottomLeftNode.val == bottomRightNode.val):
                return Node(topLeftNode.val, True, None, None, None, None)
            return Node(True, False, topLeftNode, topRightNode, bottomLeftNode, bottomRightNode)
        if not grid:
            return None
        return dfs(grid, 0, 0, len(grid))