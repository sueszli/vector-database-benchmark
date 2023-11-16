"""
Unival Trees

A unival tree (which stands for "universal value") is a tree where all nodes under it have the same value.
Given the root to a binary tree, count the number of unival subtrees.

Input:
   0
  /  1   0
    /    1   0
  /  1   1
Output: 5

=========================================
Simple tree traversal solution.
    Time Complexity:    O(N)
    Space Complexity:   O(N)    , because of the recursion stack (but this is if the tree is one branch), O(LogN) if the tree is balanced.
"""
from tree_helpers import TreeNode

def count_unival_trees(tree):
    if False:
        i = 10
        return i + 15
    if tree is None:
        return 0
    return total_unival_trees(tree)[0]

def total_unival_trees(node):
    if False:
        return 10
    left_value = None
    is_left_unival_tree = True
    right_value = None
    is_right_unival_tree = True
    unival_trees = 0
    if node.left is not None:
        left_result = total_unival_trees(node.left)
        unival_trees += left_result[0]
        is_left_unival_tree = left_result[1]
        left_value = node.left.val
    if node.right is not None:
        right_result = total_unival_trees(node.right)
        unival_trees += right_result[0]
        is_right_unival_tree = right_result[1]
        right_value = node.right.val
    is_this_unival_tree = is_left_unival_tree and is_right_unival_tree and (left_value == right_value)
    unival_trees += is_this_unival_tree
    return (unival_trees, is_this_unival_tree)
print(count_unival_trees(TreeNode(0, TreeNode(1), TreeNode(0, TreeNode(1, TreeNode(1), TreeNode(1)), TreeNode(0)))))