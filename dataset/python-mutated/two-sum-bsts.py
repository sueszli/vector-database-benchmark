class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def twoSumBSTs(self, root1, root2, target):
        if False:
            print('Hello World!')
        '\n        :type root1: TreeNode\n        :type root2: TreeNode\n        :type target: int\n        :rtype: bool\n        '

        def inorder_gen(root, asc=True):
            if False:
                return 10
            (result, stack) = ([], [(root, False)])
            while stack:
                (root, is_visited) = stack.pop()
                if root is None:
                    continue
                if is_visited:
                    yield root.val
                elif asc:
                    stack.append((root.right, False))
                    stack.append((root, True))
                    stack.append((root.left, False))
                else:
                    stack.append((root.left, False))
                    stack.append((root, True))
                    stack.append((root.right, False))
        (left_gen, right_gen) = (inorder_gen(root1, True), inorder_gen(root2, False))
        (left, right) = (next(left_gen), next(right_gen))
        while left is not None and right is not None:
            if left + right < target:
                left = next(left_gen)
            elif left + right > target:
                right = next(right_gen)
            else:
                return True
        return False