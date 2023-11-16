class Solution(object):

    def sumFourDivisors(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        for num in nums:
            (facs, i) = ([], 1)
            while i * i <= num:
                if num % i:
                    i += 1
                    continue
                facs.append(i)
                if i != num // i:
                    facs.append(num // i)
                    if len(facs) > 4:
                        break
                i += 1
            if len(facs) == 4:
                result += sum(facs)
        return result
import itertools

class Solution2(object):

    def sumFourDivisors(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def factorize(x):
            if False:
                return 10
            result = []
            d = 2
            while d * d <= x:
                e = 0
                while x % d == 0:
                    x //= d
                    e += 1
                if e:
                    result.append([d, e])
                d += 1 if d == 2 else 2
            if x > 1:
                result.append([x, 1])
            return result
        result = 0
        for facs in itertools.imap(factorize, nums):
            if len(facs) == 1 and facs[0][1] == 3:
                p = facs[0][0]
                result += (p ** 4 - 1) // (p - 1)
            elif len(facs) == 2 and facs[0][1] == facs[1][1] == 1:
                (p, q) = (facs[0][0], facs[1][0])
                result += (1 + p) * (1 + q)
        return result