def quick_sort(nums, start, end):
    if False:
        while True:
            i = 10
    i = start
    j = end
    if i >= j:
        return
    key = nums[i]
    while i < j:
        while i < j and key <= nums[j]:
            print(key, nums[j], '*' * 30)
            j -= 1
        nums[i] = nums[j]
        while i < j and key >= nums[i]:
            print(key, nums[i], '*' * 30)
            i += 1
        nums[j] = nums[i]
    nums[i] = key
    quick_sort(nums, start, i - 1)
    quick_sort(nums, i + 1, end)
if __name__ == '__main__':
    nums = [3, 6, 8, 5, 2, 4, 9, 1, 7]
    quick_sort(nums, 0, len(nums) - 1)
    print('result:', nums)