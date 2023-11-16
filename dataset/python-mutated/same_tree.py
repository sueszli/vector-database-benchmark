"""
Same Tree

Given two binary trees, write a function to check if they are the same or not.
Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

Input: 1         1
      / \\       /      2   3     2   3
Output: True

Input:  1         1
       /                 2             2
Output: False

=========================================
Traverse both trees in same time and if something isn't equal, return False.
    Time Complexity:    O(N)
    Space Complexity:   O(N)    , because of the recursion stack (but this is if the tree is one branch), O(LogN) if the tree is balanced.
"""
from tree_helpers import TreeNode

def is_same_tree(p, q):
    if False:
        while True:
            i = 10
    if p is None and p == q:
        return True
    if p is None or q is None:
        return False
    if p.val != q.val:
        return False
    if not is_same_tree(p.left, q.left):
        return False
    if not is_same_tree(p.right, q.right):
        return False
    return True
p = TreeNode(1, TreeNode(2), TreeNode(3))
q = TreeNode(1, TreeNode(2), TreeNode(3))
print(is_same_tree(p, q))
p = TreeNode(1, TreeNode(2))
q = TreeNode(1, None, TreeNode(2))
print(is_same_tree(p, q))