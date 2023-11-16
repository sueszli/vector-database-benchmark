"""
Find second largest node in bst

Given the root to a binary search tree, find the second largest node in the tree.

=========================================
There are 4 possible cases (see the details in the code).
Only 1 branch is searched to the end (leaf), not the whole tree.
    Time Complexity:    O(N)        , this is the worst case when all elements are in one (the right) branch O(N), O(LogN) if the tree is balanced (balanced bst)
    Space Complexity:   O(N)        , because of the recursion stack (but this is if the tree is one branch), O(LogN) if the tree is balanced.
The second solution is simpler and it's same as find_kth_smallest_node_bst.py but K is 2.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""
from tree_helpers import TreeNode

def find_second_largest_bst_1(root):
    if False:
        i = 10
        return i + 15
    if root == None:
        return None
    return search_1(root, False)

def search_1(node, visited_left):
    if False:
        for i in range(10):
            print('nop')
    if node.right is not None:
        result = search_1(node.right, visited_left)
        if result is None:
            return node
        return result
    if visited_left:
        return node
    if node.left is not None:
        return search_1(node.left, True)
    return None

def find_second_largest_bst_2(root):
    if False:
        for i in range(10):
            print('nop')
    return search_2(root, 2)[1]

def search_2(node, k):
    if False:
        return 10
    if node == None:
        return (k, None)
    right = search_2(node.right, k)
    if right[0] == 0:
        return right
    k = right[0] - 1
    if k == 0:
        return (0, node)
    return search_2(node.left, k)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(7), TreeNode(12, TreeNode(10, TreeNode(13)))))
print(find_second_largest_bst_1(tree).val)
print(find_second_largest_bst_2(tree).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(7), TreeNode(12)))
print(find_second_largest_bst_1(tree).val)
print(find_second_largest_bst_2(tree).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)))
print(find_second_largest_bst_1(tree).val)
print(find_second_largest_bst_2(tree).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(6, None, TreeNode(7))))
print(find_second_largest_bst_1(tree).val)
print(find_second_largest_bst_2(tree).val)
tree = TreeNode(5, TreeNode(3, TreeNode(1), TreeNode(4)), TreeNode(8, TreeNode(7), TreeNode(12, TreeNode(9, None, TreeNode(10, None, TreeNode(11))))))
print(find_second_largest_bst_1(tree).val)
print(find_second_largest_bst_2(tree).val)