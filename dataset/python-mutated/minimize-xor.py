class Solution(object):

    def minimizeXor(self, num1, num2):
        if False:
            print('Hello World!')
        '\n        :type num1: int\n        :type num2: int\n        :rtype: int\n        '

        def popcount(x):
            if False:
                i = 10
                return i + 15
            return bin(x)[2:].count('1')
        (cnt1, cnt2) = (popcount(num1), popcount(num2))
        result = num1
        cnt = abs(cnt1 - cnt2)
        expect = 1 if cnt1 >= cnt2 else 0
        i = 0
        while cnt:
            if num1 >> i & 1 == expect:
                cnt -= 1
                result ^= 1 << i
            i += 1
        return result