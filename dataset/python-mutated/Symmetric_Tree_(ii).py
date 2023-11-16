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

    def isSymmetric(self, root) -> bool:
        if False:
            print('Hello World!')
        if root.left == None and root.right == None:
            return True
        elif root.left == None:
            return False
        elif root.right == None:
            return False
        q = queue.Queue()
        q.put(root.left)
        q.put(root.right)
        while not q.empty():
            right_root = q.get()
            left_root = q.get()
            if right_root == None and left_root == None:
                continue
            elif right_root == None:
                return False
            elif left_root == None:
                return False
            elif right_root.val != left_root.val:
                return False
            q.put(left_root.left)
            q.put(right_root.right)
            q.put(left_root.right)
            q.put(right_root.left)
        return True