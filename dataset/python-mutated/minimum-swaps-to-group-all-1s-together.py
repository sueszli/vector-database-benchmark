class Solution(object):

    def minSwaps(self, data):
        if False:
            i = 10
            return i + 15
        '\n        :type data: List[int]\n        :rtype: int\n        '
        total_count = sum(data)
        (result, count, left) = (0, 0, 0)
        for i in xrange(len(data)):
            count += data[i]
            if i - left + 1 > total_count:
                count -= data[left]
                left += 1
            result = max(result, count)
        return total_count - result