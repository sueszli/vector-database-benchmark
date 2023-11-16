class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def postorderTraversal(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '
        dummy = TreeNode(0)
        dummy.left = root
        (result, cur) = ([], dummy)
        while cur:
            if cur.left is None:
                cur = cur.right
            else:
                node = cur.left
                while node.right and node.right != cur:
                    node = node.right
                if node.right is None:
                    node.right = cur
                    cur = cur.left
                else:
                    result += self.traceBack(cur.left, node)
                    node.right = None
                    cur = cur.right
        return result

    def traceBack(self, frm, to):
        if False:
            while True:
                i = 10
        (result, cur) = ([], frm)
        while cur is not to:
            result.append(cur.val)
            cur = cur.right
        result.append(to.val)
        result.reverse()
        return result

class Solution2(object):

    def postorderTraversal(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '
        (result, stack) = ([], [(root, False)])
        while stack:
            (root, is_visited) = stack.pop()
            if root is None:
                continue
            if is_visited:
                result.append(root.val)
            else:
                stack.append((root, True))
                stack.append((root.right, False))
                stack.append((root.left, False))
        return result