class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def isCousins(self, root, x, y):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type x: int\n        :type y: int\n        :rtype: bool\n        '

        def dfs(root, x, depth, parent):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return False
            if root.val == x:
                return True
            depth[0] += 1
            (prev_parent, parent[0]) = (parent[0], root)
            if dfs(root.left, x, depth, parent):
                return True
            parent[0] = root
            if dfs(root.right, x, depth, parent):
                return True
            parent[0] = prev_parent
            depth[0] -= 1
            return False
        (depth_x, depth_y) = ([0], [0])
        (parent_x, parent_y) = ([None], [None])
        return dfs(root, x, depth_x, parent_x) and dfs(root, y, depth_y, parent_y) and (depth_x[0] == depth_y[0]) and (parent_x[0] != parent_y[0])