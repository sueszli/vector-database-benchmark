import collections

class Solution(object):

    def majorityElement(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def boyer_moore_majority_vote():
            if False:
                print('Hello World!')
            (result, cnt) = (None, 0)
            for x in nums:
                if not cnt:
                    result = x
                if x == result:
                    cnt += 1
                else:
                    cnt -= 1
            return result
        return boyer_moore_majority_vote()
import collections

class Solution2(object):

    def majorityElement(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return collections.Counter(nums).most_common(1)[0][0]
import collections

class Solution3(object):

    def majorityElement(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return sorted(collections.Counter(nums).items(), key=lambda a: a[1], reverse=True)[0][0]