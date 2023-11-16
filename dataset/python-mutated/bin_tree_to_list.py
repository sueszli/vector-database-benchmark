from tree.tree import TreeNode

def bin_tree_to_list(root):
    if False:
        for i in range(10):
            print('nop')
    '\n    type root: root class\n    '
    if not root:
        return root
    root = bin_tree_to_list_util(root)
    while root.left:
        root = root.left
    return root

def bin_tree_to_list_util(root):
    if False:
        i = 10
        return i + 15
    if not root:
        return root
    if root.left:
        left = bin_tree_to_list_util(root.left)
        while left.right:
            left = left.right
        left.right = root
        root.left = left
    if root.right:
        right = bin_tree_to_list_util(root.right)
        while right.left:
            right = right.left
        right.left = root
        root.right = right
    return root

def print_tree(root):
    if False:
        for i in range(10):
            print('nop')
    while root:
        print(root.val)
        root = root.right