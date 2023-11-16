class Solution(object):

    def fizzBuzz(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: List[str]\n        '
        result = []
        for i in xrange(1, n + 1):
            if i % 15 == 0:
                result.append('FizzBuzz')
            elif i % 5 == 0:
                result.append('Buzz')
            elif i % 3 == 0:
                result.append('Fizz')
            else:
                result.append(str(i))
        return result

    def fizzBuzz2(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: List[str]\n        '
        l = [str(x) for x in range(n + 1)]
        l3 = range(0, n + 1, 3)
        l5 = range(0, n + 1, 5)
        for i in l3:
            l[i] = 'Fizz'
        for i in l5:
            if l[i] == 'Fizz':
                l[i] += 'Buzz'
            else:
                l[i] = 'Buzz'
        return l[1:]

    def fizzBuzz3(self, n):
        if False:
            for i in range(10):
                print('nop')
        return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n + 1)]

    def fizzBuzz4(self, n):
        if False:
            return 10
        return ['FizzBuzz'[i % -3 & -4:i % -5 & 8 ^ 12] or repr(i) for i in range(1, n + 1)]