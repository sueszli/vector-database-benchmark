"""
Find max path sum

Wrie a function that takes a Binary Tree and returns its max path sum.

Input:
        1
       /       2   3
     / \\ /     4  5 6  7
Output: 18
Output explanation: 5 -> 2 -> 1 -> 3 -> 7

Input:
       -1
       /      -2   3
     / \\ /    -4 -5 2  5
Output: 10
Output explanation: 2 -> 3 -> 5

=========================================
Traverse the tree and in each node compare create a new path where the left and right max subpaths are merging in the current node.
    Time Complexity:    O(N)
    Space Complexity:   O(N)        , because of the recursion stack (but this is the tree is one branch), O(LogN) if the tree is balanced.
"""
from tree_helpers import TreeNode

def max_path_sum(tree):
    if False:
        return 10
    return find_max_path_sum(tree)[0]

def find_max_path_sum(node):
    if False:
        i = 10
        return i + 15
    if node is None:
        return (0, 0)
    left_result = find_max_path_sum(node.left)
    right_result = find_max_path_sum(node.right)
    current_path = left_result[1] + node.val + right_result[1]
    max_path = max(left_result[0], current_path, right_result[0])
    max_subpath = max(left_result[1] + node.val, right_result[1] + node.val, node.val, 0)
    return (max_path, max_subpath)
tree = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, TreeNode(6), TreeNode(7)))
print(max_path_sum(tree))
tree = TreeNode(-1, TreeNode(-2, TreeNode(-4), TreeNode(-5)), TreeNode(3, TreeNode(2), TreeNode(5)))
print(max_path_sum(tree))
'\n        1\n       /       7   3\n     / \\ /    -4 -5 6  2\n'
tree = TreeNode(1, TreeNode(7, TreeNode(-4), TreeNode(-5)), TreeNode(3, TreeNode(6), TreeNode(2)))
print(max_path_sum(tree))
'\n        1\n       /       2   3\n     / \\ /    -4 -5 -2 -3\n'
tree = TreeNode(1, TreeNode(2, TreeNode(-4), TreeNode(-5)), TreeNode(3, TreeNode(-2), TreeNode(-3)))
print(max_path_sum(tree))