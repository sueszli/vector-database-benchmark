import collections

class Solution(object):

    def fourSum(self, nums, target):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[List[int]]\n        '
        nums.sort()
        result = []
        for i in xrange(len(nums) - 3):
            if i and nums[i] == nums[i - 1]:
                continue
            for j in xrange(i + 1, len(nums) - 2):
                if j != i + 1 and nums[j] == nums[j - 1]:
                    continue
                total = target - nums[i] - nums[j]
                (left, right) = (j + 1, len(nums) - 1)
                while left < right:
                    if nums[left] + nums[right] == total:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        right -= 1
                        left += 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
                    elif nums[left] + nums[right] > total:
                        right -= 1
                    else:
                        left += 1
        return result

class Solution2(object):

    def fourSum(self, nums, target):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[List[int]]\n        '
        (nums, result, lookup) = (sorted(nums), [], collections.defaultdict(list))
        for i in xrange(0, len(nums) - 1):
            for j in xrange(i + 1, len(nums)):
                is_duplicated = False
                for [x, y] in lookup[nums[i] + nums[j]]:
                    if nums[x] == nums[i]:
                        is_duplicated = True
                        break
                if not is_duplicated:
                    lookup[nums[i] + nums[j]].append([i, j])
        ans = {}
        for c in xrange(2, len(nums)):
            for d in xrange(c + 1, len(nums)):
                if target - nums[c] - nums[d] in lookup:
                    for [a, b] in lookup[target - nums[c] - nums[d]]:
                        if b < c:
                            quad = [nums[a], nums[b], nums[c], nums[d]]
                            quad_hash = ' '.join(str(quad))
                            if quad_hash not in ans:
                                ans[quad_hash] = True
                                result.append(quad)
        return result

class Solution3(object):

    def fourSum(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[List[int]]\n        '
        (nums, result, lookup) = (sorted(nums), [], collections.defaultdict(list))
        for i in xrange(0, len(nums) - 1):
            for j in xrange(i + 1, len(nums)):
                lookup[nums[i] + nums[j]].append([i, j])
        for i in lookup.keys():
            if target - i in lookup:
                for x in lookup[i]:
                    for y in lookup[target - i]:
                        ([a, b], [c, d]) = (x, y)
                        if a is not c and a is not d and (b is not c) and (b is not d):
                            quad = sorted([nums[a], nums[b], nums[c], nums[d]])
                            if quad not in result:
                                result.append(quad)
        return sorted(result)