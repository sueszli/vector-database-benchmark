class Solution(object):

    def reversePairs(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def merge(nums, start, mid, end):
            if False:
                i = 10
                return i + 15
            r = mid + 1
            tmp = []
            for i in xrange(start, mid + 1):
                while r <= end and nums[i] > nums[r]:
                    tmp.append(nums[r])
                    r += 1
                tmp.append(nums[i])
            nums[start:start + len(tmp)] = tmp

        def countAndMergeSort(nums, start, end):
            if False:
                return 10
            if end - start <= 0:
                return 0
            mid = start + (end - start) / 2
            count = countAndMergeSort(nums, start, mid) + countAndMergeSort(nums, mid + 1, end)
            r = mid + 1
            for i in xrange(start, mid + 1):
                while r <= end and nums[i] > nums[r] * 2:
                    r += 1
                count += r - (mid + 1)
            merge(nums, start, mid, end)
            return count
        return countAndMergeSort(nums, 0, len(nums) - 1)