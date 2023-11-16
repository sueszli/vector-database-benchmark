class Solution(object):

    def plusOne(self, digits):
        if False:
            print('Hello World!')
        '\n        :type digits: List[int]\n        :rtype: List[int]\n        '
        for i in reversed(xrange(len(digits))):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                return digits
        digits[0] = 1
        digits.append(0)
        return digits

class Solution2(object):

    def plusOne(self, digits):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type digits: List[int]\n        :rtype: List[int]\n        '
        result = digits[::-1]
        carry = 1
        for i in xrange(len(result)):
            result[i] += carry
            (carry, result[i]) = divmod(result[i], 10)
        if carry:
            result.append(carry)
        return result[::-1]