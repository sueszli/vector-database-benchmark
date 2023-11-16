class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def delNodes(self, root, to_delete):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :type to_delete: List[int]\n        :rtype: List[TreeNode]\n        '

        def delNodesHelper(to_delete_set, root, is_root, result):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return None
            is_deleted = root.val in to_delete_set
            if is_root and (not is_deleted):
                result.append(root)
            root.left = delNodesHelper(to_delete_set, root.left, is_deleted, result)
            root.right = delNodesHelper(to_delete_set, root.right, is_deleted, result)
            return None if is_deleted else root
        result = []
        to_delete_set = set(to_delete)
        delNodesHelper(to_delete_set, root, True, result)
        return result