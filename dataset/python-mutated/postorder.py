"""
Time complexity : O(n)
"""

class Node:

    def __init__(self, val, left=None, right=None):
        if False:
            print('Hello World!')
        self.val = val
        self.left = left
        self.right = right

def postorder(root):
    if False:
        return 10
    res_temp = []
    res = []
    if not root:
        return res
    stack = []
    stack.append(root)
    while stack:
        root = stack.pop()
        res_temp.append(root.val)
        if root.left:
            stack.append(root.left)
        if root.right:
            stack.append(root.right)
    while res_temp:
        res.append(res_temp.pop())
    return res

def postorder_rec(root, res=None):
    if False:
        i = 10
        return i + 15
    if root is None:
        return []
    if res is None:
        res = []
    postorder_rec(root.left, res)
    postorder_rec(root.right, res)
    res.append(root.val)
    return res