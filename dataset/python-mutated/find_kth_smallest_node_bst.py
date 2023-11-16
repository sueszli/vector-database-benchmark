"""
Kth Smallest Element in a BST

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it

Input: 3    , k = 1
      /      1   4
             2
Output: 1

Input: 5    , k = 3
      /      3   6
    /    2   4
  /
 1
Output: 3

=========================================
Traverse Inorder the tree (Type of depth first traversal: left, root, right) and count the nodes.
When the Kth node is found, return that node.
    Time Complexity:    O(N)
    Space Complexity:   O(N)        , because of the recursion stack (but this is if the tree is one branch), O(LogN) if the tree is balanced.
"""
from tree_helpers import TreeNode

def find_kth_smallest_node_bst(root, k):
    if False:
        while True:
            i = 10
    return search(root, k)[1]

def search(node, k):
    if False:
        return 10
    if node is None:
        return (k, None)
    left = search(node.left, k)
    if left[0] == 0:
        return left
    k = left[0] - 1
    if k == 0:
        return (k, node)
    return search(node.right, k)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(7), TreeNode(12, TreeNode(10, TreeNode(9)))))
print(find_kth_smallest_node_bst(tree, 7).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(7), TreeNode(12)))
print(find_kth_smallest_node_bst(tree, 4).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)))
print(find_kth_smallest_node_bst(tree, 2).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(6, None, TreeNode(7))))
print(find_kth_smallest_node_bst(tree, 5).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(7), TreeNode(12, TreeNode(9, None, TreeNode(10, None, TreeNode(11))))))
print(find_kth_smallest_node_bst(tree, 7).val)