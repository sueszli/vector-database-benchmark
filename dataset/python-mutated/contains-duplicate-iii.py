import collections

class Solution(object):

    def containsNearbyAlmostDuplicate(self, nums, k, t):
        if False:
            for i in range(10):
                print('nop')
        if k < 0 or t < 0:
            return False
        window = collections.OrderedDict()
        for n in nums:
            if len(window) > k:
                window.popitem(False)
            bucket = n if not t else n // t
            for m in (window.get(bucket - 1), window.get(bucket), window.get(bucket + 1)):
                if m is not None and abs(n - m) <= t:
                    return True
            window[bucket] = n
        return False