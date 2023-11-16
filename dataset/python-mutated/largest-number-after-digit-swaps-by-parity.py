class Solution(object):

    def largestInteger(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: int\n        '

        def count(num):
            if False:
                while True:
                    i = 10
            cnt = [0] * 10
            while num:
                (num, d) = divmod(num, 10)
                cnt[d] += 1
            return cnt
        cnt = count(num)
        result = 0
        digit = [0, 1]
        base = 1
        while num:
            (num, d) = divmod(num, 10)
            while not cnt[digit[d % 2]]:
                digit[d % 2] += 2
            cnt[digit[d % 2]] -= 1
            result += digit[d % 2] * base
            base *= 10
        return result