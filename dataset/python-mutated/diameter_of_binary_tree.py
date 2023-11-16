"""
Diameter of Binary Tree

Given a binary tree, you need to compute the length of the diameter of the tree.
The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
This path may or may not pass through the root.
Note: The length of path between two nodes is represented by the number of nodes.

Input: 3
      /      1   4
      \\          2   5
      /
     7
Output: 6
Output Explanation: [7, 2, 1, 3, 4, 5] is the diameter of the binary tree.

Input: 5
      /      3   6
    /    2   4
  /      1       8
Output: 5
Output Explanation: [1, 2, 3, 4, 8] is the diameter of the binary tree.

=========================================
Traverse the tree and keep/return information about the longest/max branch and longest/max diameter.
    Time Complexity:    O(N)
    Space Complexity:   O(N)        , because of the recursion stack (but this is if the tree is one branch), O(LogN) if the tree is balanced.
"""
from tree_helpers import TreeNode

def diameter(root):
    if False:
        for i in range(10):
            print('nop')
    return find_diameter(root)[1]

def find_diameter(root):
    if False:
        print('Hello World!')
    ' returns (max branch length, max diameter) '
    if not root:
        return (0, 0)
    (left, right) = (find_diameter(root.left), find_diameter(root.right))
    return (max(left[0], right[0]) + 1, max(left[1], right[1], left[0] + right[0] + 1))
tree = TreeNode(3, TreeNode(1, None, TreeNode(2, TreeNode(7))), TreeNode(4, None, TreeNode(5)))
print(diameter(tree))
tree = TreeNode(5, TreeNode(3, TreeNode(2, TreeNode(1)), TreeNode(4, None, TreeNode(8))), TreeNode(6))
print(diameter(tree))