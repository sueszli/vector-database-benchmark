"""
Find second largest node (not search tree)

Given the root to a tree (not bst), find the second largest node in the tree.

=========================================
Traverse tree and compare the current value with the saved 2 values.
    Time Complexity:    O(N)
    Space Complexity:   O(N)        , because of the recursion stack (but this is the tree is one branch), O(LogN) if the tree is balanced.
"""
from tree_helpers import TreeNode
import math

def find_second_largest(root):
    if False:
        for i in range(10):
            print('nop')
    arr = [TreeNode(-math.inf), TreeNode(-math.inf)]
    traverse_tree(root, arr)
    if arr[1] == -math.inf:
        return None
    return arr[1]

def traverse_tree(node, arr):
    if False:
        i = 10
        return i + 15
    if node == None:
        return
    if arr[0].val < node.val:
        arr[1] = arr[0]
        arr[0] = node
    elif arr[1].val < node.val:
        arr[1] = node
    traverse_tree(node.left, arr)
    traverse_tree(node.right, arr)
tree = TreeNode(1, TreeNode(5, TreeNode(2), TreeNode(8)), TreeNode(4, TreeNode(12), TreeNode(7)))
print(find_second_largest(tree).val)