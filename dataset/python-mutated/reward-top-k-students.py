import random
import itertools

class Solution(object):

    def topStudents(self, positive_feedback, negative_feedback, report, student_id, k):
        if False:
            while True:
                i = 10
        '\n        :type positive_feedback: List[str]\n        :type negative_feedback: List[str]\n        :type report: List[str]\n        :type student_id: List[int]\n        :type k: int\n        :rtype: List[int]\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                return 10

            def tri_partition(nums, left, right, target, compare):
                if False:
                    while True:
                        i = 10
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
                pivot_idx = random.randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1
        (pos, neg) = (set(positive_feedback), set(negative_feedback))
        arr = []
        for (i, r) in itertools.izip(student_id, report):
            score = sum((3 if w in pos else -1 if w in neg else 0 for w in r.split()))
            arr.append((-score, i))
        nth_element(arr, k - 1)
        return [i for (_, i) in sorted(arr[:k])]