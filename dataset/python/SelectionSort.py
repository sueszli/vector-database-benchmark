# coding:utf8
"""
选择排序和冒泡排序的区别在于：

选择排序的前提是：找到最小值的位置，最后才进行1次交换
而冒泡排序：相邻的值进行交换，一共进行n次交换
"""
def selection_sort(nums):
    for i in range(len(nums)-1):
        index = i
        # 考虑到数组会遇到多个最小值，所以比较的时候直接用index表示当前比较最小
        for j in range(i+1, len(nums)):
            if nums[index] > nums[j]:
                index = j
        nums[i], nums[index] = nums[index], nums[i]


if __name__ == "__main__":
    nums = [3, 6, 8, 5, 2, 4, 9, 1, 7]
    selection_sort(nums)
    print('result:', nums)
