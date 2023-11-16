def closest_value(root, target):
    if False:
        for i in range(10):
            print('nop')
    '\n    :type root: TreeNode\n    :type target: float\n    :rtype: int\n    '
    a = root.val
    kid = root.left if target < a else root.right
    if not kid:
        return a
    b = closest_value(kid, target)
    return min((a, b), key=lambda x: abs(target - x))