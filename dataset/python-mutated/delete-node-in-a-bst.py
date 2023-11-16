class Solution(object):

    def deleteNode(self, root, key):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :type key: int\n        :rtype: TreeNode\n        '
        if not root:
            return root
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        elif not root.left:
            right = root.right
            del root
            return right
        elif not root.right:
            left = root.left
            del root
            return left
        else:
            successor = root.right
            while successor.left:
                successor = successor.left
            root.val = successor.val
            root.right = self.deleteNode(root.right, successor.val)
        return root