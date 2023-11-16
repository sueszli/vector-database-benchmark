"""
Given an array of n integers, are there elements a, b, .. , n in nums
such that a + b + .. + n = target?

Find all unique n-tuplets in the array which gives the sum of target.

Example:
    basic:
        Given:
            n = 4
            nums = [1, 0, -1, 0, -2, 2]
            target = 0,
        return [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]

    advanced:
        Given:
            n = 2
            nums = [[-3, 0], [-2, 1], [2, 2], [3, 3], [8, 4], [-9, 5]]
            target = -5
            def sum(a, b):
                return [a[0] + b[1], a[1] + b[0]]
            def compare(num, target):
                if num[0] < target:
                    return -1
                elif if num[0] > target:
                    return 1
                else:
                    return 0
        return [[-9, 5], [8, 4]]
(TL:DR) because -9 + 4 = -5
"""

def n_sum(n, nums, target, **kv):
    if False:
        while True:
            i = 10
    "\n    n: int\n    nums: list[object]\n    target: object\n    sum_closure: function, optional\n        Given two elements of nums, return sum of both.\n    compare_closure: function, optional\n        Given one object of nums and target, return -1, 1, or 0.\n    same_closure: function, optional\n        Given two object of nums, return bool.\n    return: list[list[object]]\n\n    Note:\n    1. type of sum_closure's return should be same \n       as type of compare_closure's first param\n    "

    def sum_closure_default(a, b):
        if False:
            return 10
        return a + b

    def compare_closure_default(num, target):
        if False:
            return 10
        ' above, below, or right on? '
        if num < target:
            return -1
        elif num > target:
            return 1
        else:
            return 0

    def same_closure_default(a, b):
        if False:
            while True:
                i = 10
        return a == b

    def n_sum(n, nums, target):
        if False:
            for i in range(10):
                print('nop')
        if n == 2:
            results = two_sum(nums, target)
        else:
            results = []
            prev_num = None
            for (index, num) in enumerate(nums):
                if prev_num is not None and same_closure(prev_num, num):
                    continue
                prev_num = num
                n_minus1_results = n_sum(n - 1, nums[index + 1:], target - num)
                n_minus1_results = append_elem_to_each_list(num, n_minus1_results)
                results += n_minus1_results
        return union(results)

    def two_sum(nums, target):
        if False:
            return 10
        nums.sort()
        lt = 0
        rt = len(nums) - 1
        results = []
        while lt < rt:
            sum_ = sum_closure(nums[lt], nums[rt])
            flag = compare_closure(sum_, target)
            if flag == -1:
                lt += 1
            elif flag == 1:
                rt -= 1
            else:
                results.append(sorted([nums[lt], nums[rt]]))
                lt += 1
                rt -= 1
                while lt < len(nums) and same_closure(nums[lt - 1], nums[lt]):
                    lt += 1
                while 0 <= rt and same_closure(nums[rt], nums[rt + 1]):
                    rt -= 1
        return results

    def append_elem_to_each_list(elem, container):
        if False:
            for i in range(10):
                print('nop')
        results = []
        for elems in container:
            elems.append(elem)
            results.append(sorted(elems))
        return results

    def union(duplicate_results):
        if False:
            for i in range(10):
                print('nop')
        results = []
        if len(duplicate_results) != 0:
            duplicate_results.sort()
            results.append(duplicate_results[0])
            for result in duplicate_results[1:]:
                if results[-1] != result:
                    results.append(result)
        return results
    sum_closure = kv.get('sum_closure', sum_closure_default)
    same_closure = kv.get('same_closure', same_closure_default)
    compare_closure = kv.get('compare_closure', compare_closure_default)
    nums.sort()
    return n_sum(n, nums, target)