class Solution(object):

    def rearrangeArray(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                print('Hello World!')

            def tri_partition(nums, left, right, target, compare):
                if False:
                    for i in range(10):
                        print('nop')
                mid = left
                while mid <= right:
                    if nums[mid] == target:
                        mid += 1
                    elif compare(nums[mid], target):
                        (nums[left], nums[mid]) = (nums[mid], nums[left])
                        left += 1
                        mid += 1
                    else:
                        (nums[mid], nums[right]) = (nums[right], nums[mid])
                        right -= 1
                return (left, right)
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                pivot_idx = randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1

        def reversedTriPartitionWithVI(nums, val):
            if False:
                for i in range(10):
                    print('nop')

            def idx(i, N):
                if False:
                    i = 10
                    return i + 15
                return (1 + 2 * i) % N
            N = len(nums) // 2 * 2 + 1
            (i, j, n) = (0, 0, len(nums) - 1)
            while j <= n:
                if nums[idx(j, N)] > val:
                    (nums[idx(i, N)], nums[idx(j, N)]) = (nums[idx(j, N)], nums[idx(i, N)])
                    i += 1
                    j += 1
                elif nums[idx(j, N)] < val:
                    (nums[idx(j, N)], nums[idx(n, N)]) = (nums[idx(n, N)], nums[idx(j, N)])
                    n -= 1
                else:
                    j += 1
        mid = (len(nums) - 1) // 2
        nth_element(nums, mid)
        reversedTriPartitionWithVI(nums, nums[mid])
        return nums

class Solution2(object):

    def rearrangeArray(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        nums.sort()
        mid = (len(nums) - 1) // 2
        (nums[::2], nums[1::2]) = (nums[mid::-1], nums[:mid:-1])
        return nums