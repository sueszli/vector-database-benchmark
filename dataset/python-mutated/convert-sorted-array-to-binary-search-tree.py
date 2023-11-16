class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def sortedArrayToBST(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: TreeNode\n        '
        return self.sortedArrayToBSTRecu(nums, 0, len(nums))

    def sortedArrayToBSTRecu(self, nums, start, end):
        if False:
            print('Hello World!')
        if start == end:
            return None
        mid = start + self.perfect_tree_pivot(end - start)
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBSTRecu(nums, start, mid)
        node.right = self.sortedArrayToBSTRecu(nums, mid + 1, end)
        return node

    def perfect_tree_pivot(self, n):
        if False:
            i = 10
            return i + 15
        '\n        Find the point to partition n keys for a perfect binary search tree\n        '
        x = 1
        x = 1 << n.bit_length() - 1
        if x // 2 - 1 <= n - x:
            return x - 1
        else:
            return n - x // 2

class Solution2(object):

    def sortedArrayToBST(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: TreeNode\n        '
        self.iterator = iter(nums)
        return self.helper(0, len(nums))

    def helper(self, start, end):
        if False:
            while True:
                i = 10
        if start == end:
            return None
        mid = (start + end) // 2
        left = self.helper(start, mid)
        current = TreeNode(next(self.iterator))
        current.left = left
        current.right = self.helper(mid + 1, end)
        return current