class Solution(object):

    def maxNumber(self, nums1, nums2, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type k: int\n        :rtype: List[int]\n        '

        def get_max_digits(nums, start, end, max_digits):
            if False:
                print('Hello World!')
            max_digits[end] = max_digit(nums, end)
            for i in reversed(xrange(start, end)):
                max_digits[i] = delete_digit(max_digits[i + 1])

        def max_digit(nums, k):
            if False:
                for i in range(10):
                    print('nop')
            drop = len(nums) - k
            res = []
            for num in nums:
                while drop and res and (res[-1] < num):
                    res.pop()
                    drop -= 1
                res.append(num)
            return res[:k]

        def delete_digit(nums):
            if False:
                i = 10
                return i + 15
            res = list(nums)
            for i in xrange(len(res)):
                if i == len(res) - 1 or res[i] < res[i + 1]:
                    res = res[:i] + res[i + 1:]
                    break
            return res

        def merge(a, b):
            if False:
                while True:
                    i = 10
            return [max(a, b).pop(0) for _ in xrange(len(a) + len(b))]
        (m, n) = (len(nums1), len(nums2))
        (max_digits1, max_digits2) = ([[] for _ in xrange(k + 1)], [[] for _ in xrange(k + 1)])
        get_max_digits(nums1, max(0, k - n), min(k, m), max_digits1)
        get_max_digits(nums2, max(0, k - m), min(k, n), max_digits2)
        return max((merge(max_digits1[i], max_digits2[k - i]) for i in xrange(max(0, k - n), min(k, m) + 1)))