class Solution:
    """
    Function Description:
    Input: root of a BST and an integer
    Output: subtree of a node whose value is equal to the given integer
    """

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if False:
            return 10
        if not root:
            return None
        if root.val == val:
            return root
        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)