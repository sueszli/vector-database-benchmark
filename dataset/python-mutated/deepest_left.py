from tree.tree import TreeNode

class DeepestLeft:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.depth = 0
        self.Node = None

def find_deepest_left(root, is_left, depth, res):
    if False:
        while True:
            i = 10
    if not root:
        return
    if is_left and depth > res.depth:
        res.depth = depth
        res.Node = root
    find_deepest_left(root.left, True, depth + 1, res)
    find_deepest_left(root.right, False, depth + 1, res)
if __name__ == '__main__':
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)
    root.right.right.right = TreeNode(7)
    res = DeepestLeft()
    find_deepest_left(root, True, 1, res)
    if res.Node:
        print(res.Node.val)