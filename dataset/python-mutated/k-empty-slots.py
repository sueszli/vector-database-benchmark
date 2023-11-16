class Solution(object):

    def kEmptySlots(self, flowers, k):
        if False:
            print('Hello World!')
        '\n        :type flowers: List[int]\n        :type k: int\n        :rtype: int\n        '
        days = [0] * len(flowers)
        for i in xrange(len(flowers)):
            days[flowers[i] - 1] = i
        result = float('inf')
        (i, left, right) = (0, 0, k + 1)
        while right < len(days):
            if days[i] < days[left] or days[i] <= days[right]:
                if i == right:
                    result = min(result, max(days[left], days[right]))
                (left, right) = (i, k + 1 + i)
            i += 1
        return -1 if result == float('inf') else result + 1