import itertools

class Solution(object):

    def numOfSubarrays(self, arr, k, threshold):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :type k: int\n        :type threshold: int\n        :rtype: int\n        '
        (result, curr) = (0, sum(itertools.islice(arr, 0, k - 1)))
        for i in xrange(k - 1, len(arr)):
            curr += arr[i] - (arr[i - k] if i - k >= 0 else 0)
            result += int(curr >= threshold * k)
        return result

class Solution2(object):

    def numOfSubarrays(self, arr, k, threshold):
        if False:
            print('Hello World!')
        '\n        :type arr: List[int]\n        :type k: int\n        :type threshold: int\n        :rtype: int\n        '
        accu = [0]
        for x in arr:
            accu.append(accu[-1] + x)
        result = 0
        for i in xrange(len(accu) - k):
            if accu[i + k] - accu[i] >= threshold * k:
                result += 1
        return result