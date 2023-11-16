def is_balanced(root):
    if False:
        print('Hello World!')
    return __is_balanced_recursive(root)

def __is_balanced_recursive(root):
    if False:
        i = 10
        return i + 15
    '\n    O(N) solution\n    '
    return -1 != __get_depth(root)

def __get_depth(root):
    if False:
        print('Hello World!')
    '\n    return 0 if unbalanced else depth + 1\n    '
    if root is None:
        return 0
    left = __get_depth(root.left)
    right = __get_depth(root.right)
    if abs(left - right) > 1 or -1 in [left, right]:
        return -1
    return 1 + max(left, right)