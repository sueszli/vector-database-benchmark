class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def inorderTraversal(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '
        (result, curr) = ([], root)
        while curr:
            if curr.left is None:
                result.append(curr.val)
                curr = curr.right
            else:
                node = curr.left
                while node.right and node.right != curr:
                    node = node.right
                if node.right is None:
                    node.right = curr
                    curr = curr.left
                else:
                    result.append(curr.val)
                    node.right = None
                    curr = curr.right
        return result

class Solution2(object):

    def inorderTraversal(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '
        (result, stack) = ([], [(root, False)])
        while stack:
            (root, is_visited) = stack.pop()
            if root is None:
                continue
            if is_visited:
                result.append(root.val)
            else:
                stack.append((root.right, False))
                stack.append((root, True))
                stack.append((root.left, False))
        return result