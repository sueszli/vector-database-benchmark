class Solution(object):

    def kMirror(self, k, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type k: int\n        :type n: int\n        :rtype: int\n        '

        def mirror(n, base, odd):
            if False:
                i = 10
                return i + 15
            result = n
            if odd:
                n //= base
            while n:
                result = result * base + n % base
                n //= base
            return result

        def num_gen(base):
            if False:
                print('Hello World!')
            (prefix_num, total) = ([1] * 2, [base] * 2)
            odd = 1
            while True:
                x = mirror(prefix_num[odd], base, odd)
                prefix_num[odd] += 1
                if prefix_num[odd] == total[odd]:
                    total[odd] *= base
                    odd ^= 1
                yield x

        def reverse(n, base):
            if False:
                i = 10
                return i + 15
            result = 0
            while n:
                result = result * base + n % base
                n = n // base
            return result

        def mirror_num(gen, base):
            if False:
                return 10
            while True:
                x = next(gen)
                if x == reverse(x, base):
                    break
            return x
        (base1, base2) = (k, 10)
        gen = num_gen(base1)
        return sum((mirror_num(gen, base2) for _ in xrange(n)))

class Solution2(object):

    def kMirror(self, k, n):
        if False:
            return 10
        '\n        :type k: int\n        :type n: int\n        :rtype: int\n        '

        def num_gen(k):
            if False:
                return 10
            digits = ['0']
            while True:
                for i in xrange(len(digits) // 2, len(digits)):
                    if int(digits[i]) + 1 < k:
                        digits[i] = digits[-1 - i] = str(int(digits[i]) + 1)
                        break
                    digits[i] = digits[-1 - i] = '0'
                else:
                    digits.insert(0, '1')
                    digits[-1] = '1'
                yield ''.join(digits)

        def mirror_num(gen):
            if False:
                for i in range(10):
                    print('nop')
            while True:
                x = int(next(gen, k), k)
                if str(x) == str(x)[::-1]:
                    break
            return x
        gen = num_gen(k)
        return sum((mirror_num(gen) for _ in xrange(n)))