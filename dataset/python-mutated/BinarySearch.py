"""
1. 二分查找是有条件的，首先是有序，其次因为二分查找操作的是下标，所以要求是顺序表
2. 最优时间复杂度：O(1)
3. 最坏时间复杂度：O(logn)
"""

def binary_search(nums, data):
    if False:
        while True:
            i = 10
    '\n    递归解决二分查找: nums 是一个有序数组\n    :param nums:\n    :return:\n    '
    n = len(nums)
    if n < 1:
        return False
    mid = n // 2
    if nums[mid] > data:
        return binary_search(nums[:mid], data)
    elif nums[mid] < data:
        return binary_search(nums[mid + 1:], data)
    else:
        return True
if __name__ == '__main__':
    nums = [1, 4, 6, 8, 10, 20, 25, 30]
    if binary_search(nums, 8):
        print('ok')