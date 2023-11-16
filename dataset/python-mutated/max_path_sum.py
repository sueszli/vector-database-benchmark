def max_path_sum(root):
    if False:
        return 10
    maximum = float('-inf')
    helper(root, maximum)
    return maximum

def helper(root, maximum):
    if False:
        return 10
    if root is None:
        return 0
    left = helper(root.left, maximum)
    right = helper(root.right, maximum)
    maximum = max(maximum, left + right + root.val)
    return root.val + maximum