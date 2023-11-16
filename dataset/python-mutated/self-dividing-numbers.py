class Solution(object):

    def selfDividingNumbers(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type left: int\n        :type right: int\n        :rtype: List[int]\n        '

        def isDividingNumber(num):
            if False:
                for i in range(10):
                    print('nop')
            n = num
            while n > 0:
                (n, r) = divmod(n, 10)
                if r == 0 or num % r != 0:
                    return False
            return True
        return [num for num in xrange(left, right + 1) if isDividingNumber(num)]
import itertools

class Solution2(object):

    def selfDividingNumbers(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type left: int\n        :type right: int\n        :rtype: List[int]\n        '
        return [num for num in xrange(left, right + 1) if not any(itertools.imap(lambda x: int(x) == 0 or num % int(x) != 0, str(num)))]