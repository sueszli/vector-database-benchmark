class Solution(object):

    def earliestFullBloom(self, plantTime, growTime):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type plantTime: List[int]\n        :type growTime: List[int]\n        :rtype: int\n        '
        order = range(len(growTime))
        order.sort(key=lambda x: growTime[x], reverse=True)
        result = curr = 0
        for i in order:
            curr += plantTime[i]
            result = max(result, curr + growTime[i])
        return result