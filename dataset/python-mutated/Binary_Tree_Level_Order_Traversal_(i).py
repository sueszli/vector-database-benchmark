import queue

class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def levelOrder(self, root):
        if False:
            print('Hello World!')
        if not root:
            return []
        levels = []
        q = queue.Queue()
        q.put([root, 0])
        while not q.empty():
            top = q.get()
            while top[1] >= len(levels):
                levels.append([])
            levels[top[1]].append(top[0].val)
            if top[0].left:
                q.put([top[0].left, top[1] + 1])
            if top[0].right:
                q.put([top[0].right, top[1] + 1])
        return levels