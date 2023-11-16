import collections

class Solution(object):

    def majorityElement(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        (k, n, cnts) = (3, len(nums), collections.defaultdict(int))
        for i in nums:
            cnts[i] += 1
            if len(cnts) == k:
                for j in cnts.keys():
                    cnts[j] -= 1
                    if cnts[j] == 0:
                        del cnts[j]
        for i in cnts.keys():
            cnts[i] = 0
        for i in nums:
            if i in cnts:
                cnts[i] += 1
        result = []
        for i in cnts.keys():
            if cnts[i] > n / k:
                result.append(i)
        return result

    def majorityElement2(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return [i[0] for i in collections.Counter(nums).items() if i[1] > len(nums) / 3]