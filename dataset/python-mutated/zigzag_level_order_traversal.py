"""
Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values.
(ie, from left to right, then right to left for the next level and alternate between).

Input: 3
      /      9  20
       /        15   7
Output: [[3], [20, 9], [15, 7]]

=========================================
Breadth first (level order) traversal, using queue.
In the end reverse each odd level.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""
from tree_helpers import TreeNode
from collections import deque

class TreeNode:

    def __init__(self, val, left=None, right=None):
        if False:
            print('Hello World!')
        self.val = val
        self.left = left
        self.right = right

def zigzag_level_order_traversal(root):
    if False:
        i = 10
        return i + 15
    results = []
    queue = deque()
    queue.append((root, 0))
    while queue:
        (node, lvl) = queue.popleft()
        if node is None:
            continue
        if len(results) < lvl + 1:
            results.append([])
        results[lvl].append(node.val)
        lvl += 1
        queue.append((node.left, lvl))
        queue.append((node.right, lvl))
    for i in range(1, len(results), 2):
        results[i] = results[i][::-1]
    return results
tree = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(zigzag_level_order_traversal(tree))