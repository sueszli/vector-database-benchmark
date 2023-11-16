class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def getAllElements(self, root1, root2):
        if False:
            i = 10
            return i + 15
        '\n        :type root1: TreeNode\n        :type root2: TreeNode\n        :rtype: List[int]\n        '

        def inorder_gen(root):
            if False:
                i = 10
                return i + 15
            (result, stack) = ([], [(root, False)])
            while stack:
                (root, is_visited) = stack.pop()
                if root is None:
                    continue
                if is_visited:
                    yield root.val
                else:
                    stack.append((root.right, False))
                    stack.append((root, True))
                    stack.append((root.left, False))
            yield None
        result = []
        (left_gen, right_gen) = (inorder_gen(root1), inorder_gen(root2))
        (left, right) = (next(left_gen), next(right_gen))
        while left is not None or right is not None:
            if right is None or (left is not None and left < right):
                result.append(left)
                left = next(left_gen)
            else:
                result.append(right)
                right = next(right_gen)
        return result