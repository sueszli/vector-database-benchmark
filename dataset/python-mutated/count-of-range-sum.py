class Solution(object):

    def countRangeSum(self, nums, lower, upper):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type lower: int\n        :type upper: int\n        :rtype: int\n        '

        def countAndMergeSort(sums, start, end, lower, upper):
            if False:
                print('Hello World!')
            if end - start <= 1:
                return 0
            mid = start + (end - start) / 2
            count = countAndMergeSort(sums, start, mid, lower, upper) + countAndMergeSort(sums, mid, end, lower, upper)
            (j, k, r) = (mid, mid, mid)
            tmp = []
            for i in xrange(start, mid):
                while k < end and sums[k] - sums[i] < lower:
                    k += 1
                while j < end and sums[j] - sums[i] <= upper:
                    j += 1
                count += j - k
                while r < end and sums[r] < sums[i]:
                    tmp.append(sums[r])
                    r += 1
                tmp.append(sums[i])
            sums[start:start + len(tmp)] = tmp
            return count
        sums = [0] * (len(nums) + 1)
        for i in xrange(len(nums)):
            sums[i + 1] = sums[i] + nums[i]
        return countAndMergeSort(sums, 0, len(sums), lower, upper)

class Solution2(object):

    def countRangeSum(self, nums, lower, upper):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type lower: int\n        :type upper: int\n        :rtype: int\n        '

        def countAndMergeSort(sums, start, end, lower, upper):
            if False:
                while True:
                    i = 10
            if end - start <= 0:
                return 0
            mid = start + (end - start) / 2
            count = countAndMergeSort(sums, start, mid, lower, upper) + countAndMergeSort(sums, mid + 1, end, lower, upper)
            (j, k, r) = (mid + 1, mid + 1, mid + 1)
            tmp = []
            for i in xrange(start, mid + 1):
                while k <= end and sums[k] - sums[i] < lower:
                    k += 1
                while j <= end and sums[j] - sums[i] <= upper:
                    j += 1
                count += j - k
                while r <= end and sums[r] < sums[i]:
                    tmp.append(sums[r])
                    r += 1
                tmp.append(sums[i])
            sums[start:start + len(tmp)] = tmp
            return count
        sums = [0] * (len(nums) + 1)
        for i in xrange(len(nums)):
            sums[i + 1] = sums[i] + nums[i]
        return countAndMergeSort(sums, 0, len(sums) - 1, lower, upper)