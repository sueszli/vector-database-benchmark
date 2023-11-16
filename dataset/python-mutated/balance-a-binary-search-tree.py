class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def balanceBST(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '

        def inorderTraversal(root):
            if False:
                while True:
                    i = 10
            (result, stk) = ([], [(root, False)])
            while stk:
                (node, is_visited) = stk.pop()
                if node is None:
                    continue
                if is_visited:
                    result.append(node.val)
                else:
                    stk.append((node.right, False))
                    stk.append((node, True))
                    stk.append((node.left, False))
            return result

        def sortedArrayToBst(arr):
            if False:
                return 10
            (ROOT, LEFT, RIGHT) = range(3)
            result = [None]
            stk = [(0, len(arr), ROOT, result)]
            while stk:
                (i, j, update, ret) = stk.pop()
                if i >= j:
                    continue
                mid = i + (j - i) // 2
                node = TreeNode(arr[mid])
                if update == ROOT:
                    ret[0] = node
                elif update == LEFT:
                    ret[0].left = node
                else:
                    ret[0].right = node
                stk.append((mid + 1, j, RIGHT, [node]))
                stk.append((i, mid, LEFT, [node]))
            return result[0]
        return sortedArrayToBst(inorderTraversal(root))

class Solution2(object):

    def balanceBST(self, root):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :rtype: TreeNode\n        '

        def inorderTraversalHelper(node, arr):
            if False:
                return 10
            if not node:
                return
            inorderTraversalHelper(node.left, arr)
            arr.append(node.val)
            inorderTraversalHelper(node.right, arr)

        def sortedArrayToBstHelper(arr, i, j):
            if False:
                while True:
                    i = 10
            if i >= j:
                return None
            mid = i + (j - i) // 2
            node = TreeNode(arr[mid])
            node.left = sortedArrayToBstHelper(arr, i, mid)
            node.right = sortedArrayToBstHelper(arr, mid + 1, j)
            return node
        arr = []
        inorderTraversalHelper(root, arr)
        return sortedArrayToBstHelper(arr, 0, len(arr))