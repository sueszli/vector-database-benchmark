"""
Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST.

Basically, the deletion can be divided into two stages:

Search for a node to remove.
If the node is found, delete the node.
Note: Time complexity should be O(height of tree).

Example:

root = [5,3,6,2,4,null,7]
key = 3

    5
   /   3   6
 / \\   2   4   7

Given key to delete is 3. So we find the node with value 3 and delete it.

One valid answer is [5,4,6,2,null,null,7], shown in the following BST.

    5
   /   4   6
 /     2       7

Another valid answer is [5,2,6,null,4,null,7].

    5
   /   2   6
   \\       4   7
"""

class Solution(object):

    def delete_node(self, root, key):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :type key: int\n        :rtype: TreeNode\n        '
        if not root:
            return None
        if root.val == key:
            if root.left:
                left_right_most = root.left
                while left_right_most.right:
                    left_right_most = left_right_most.right
                left_right_most.right = root.right
                return root.left
            else:
                return root.right
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
        return root