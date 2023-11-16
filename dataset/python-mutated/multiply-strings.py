class Solution(object):

    def multiply(self, num1, num2):
        if False:
            while True:
                i = 10
        '\n        :type num1: str\n        :type num2: str\n        :rtype: str\n        '
        result = [0] * (len(num1) + len(num2))
        for i in reversed(xrange(len(num1))):
            for j in reversed(xrange(len(num2))):
                result[i + j + 1] += int(num1[i]) * int(num2[j])
                result[i + j] += result[i + j + 1] // 10
                result[i + j + 1] %= 10
        for i in xrange(len(result)):
            if result[i]:
                break
        return ''.join(map(lambda x: str(x), result[i:]))

class Solution2(object):

    def multiply(self, num1, num2):
        if False:
            print('Hello World!')
        '\n        :type num1: str\n        :type num2: str\n        :rtype: str\n        '
        (num1, num2) = (num1[::-1], num2[::-1])
        result = [0] * (len(num1) + len(num2))
        for i in xrange(len(num1)):
            for j in xrange(len(num2)):
                result[i + j] += int(num1[i]) * int(num2[j])
                result[i + j + 1] += result[i + j] // 10
                result[i + j] %= 10
        for i in reversed(xrange(len(result))):
            if result[i]:
                break
        return ''.join(map(str, result[i::-1]))

class Solution3(object):

    def multiply(self, num1, num2):
        if False:
            while True:
                i = 10
        '\n        :type num1: str\n        :type num2: str\n        :rtype: str\n        '
        return str(int(num1) * int(num2))