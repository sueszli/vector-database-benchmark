class Solution(object):

    def closestValue(self, root, target):
        if False:
            while True:
                i = 10
        '\n        :type root: TreeNode\n        :type target: float\n        :rtype: int\n        '
        gap = float('inf')
        closest = float('inf')
        while root:
            if abs(root.val - target) < gap:
                gap = abs(root.val - target)
                closest = root.val
            if target == root.val:
                break
            elif target < root.val:
                root = root.left
            else:
                root = root.right
        return closest