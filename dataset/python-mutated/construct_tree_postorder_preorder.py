"""
    Given two arrays representing preorder and postorder traversal of a full
    binary tree, construct the binary tree and print the inorder traversal of the
    tree.
    A full binary tree has either 0 or 2 children.
    Algorithm:
        1. Assign the first element of preorder array as root of the tree.
        2. Find the same element in the postorder array and divide the postorder
            array into left and right subtree.
        3. Repeat the above steps for all the elements and construct the tree.
    Eg: pre[] = {1, 2, 4, 8, 9, 5, 3, 6, 7}
        post[] = {8, 9, 4, 5, 2, 6, 7, 3, 1}
        Tree:
                1
              /                2     3
            / \\   /            4   5 6   7
          /          8   9
      Output: 8 4 9 2 5 1 6 3 7
"""

class TreeNode:

    def __init__(self, val, left=None, right=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.left = left
        self.right = right
pre_index = 0

def construct_tree_util(pre: list, post: list, low: int, high: int, size: int):
    if False:
        for i in range(10):
            print('nop')
    '\n        Recursive function that constructs tree from preorder and postorder array.\n        \n        preIndex is a global variable that keeps track of the index in preorder\n        array.\n        preorder and postorder array are represented are pre[] and post[] respectively.\n        low and high are the indices for the postorder array.\n    '
    global pre_index
    if pre_index == -1:
        pre_index = 0
    if pre_index >= size or low > high:
        return None
    root = TreeNode(pre[pre_index])
    pre_index += 1
    if low == high or pre_index >= size:
        return root
    i = low
    while i <= high:
        if pre[pre_index] == post[i]:
            break
        i += 1
    if i <= high:
        root.left = construct_tree_util(pre, post, low, i, size)
        root.right = construct_tree_util(pre, post, i + 1, high, size)
    return root

def construct_tree(pre: list, post: list, size: int):
    if False:
        i = 10
        return i + 15
    '\n        Main Function that will construct the full binary tree from given preorder\n        and postorder array.\n    '
    global pre_index
    root = construct_tree_util(pre, post, 0, size - 1, size)
    return print_inorder(root)

def print_inorder(root: TreeNode, result=None):
    if False:
        while True:
            i = 10
    '\n        Prints the tree constructed in inorder format\n    '
    if root is None:
        return []
    if result is None:
        result = []
    print_inorder(root.left, result)
    result.append(root.val)
    print_inorder(root.right, result)
    return result
if __name__ == '__main__':
    pre = [1, 2, 4, 5, 3, 6, 7]
    post = [4, 5, 2, 6, 7, 3, 1]
    size = len(pre)
    result = construct_tree(pre, post, size)
    print(result)