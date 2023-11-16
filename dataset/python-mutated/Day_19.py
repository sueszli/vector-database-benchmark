class AdvancedArithmetic(object):

    def divisorSum(n):
        if False:
            return 10
        raise NotImplementedError

class Calculator(AdvancedArithmetic):

    def divisorSum(self, n):
        if False:
            print('Hello World!')
        p = e = 0
        for i in range(1, 1001):
            e = n % i
            if e == 0:
                p += i
        return p
        pass
n = int(input())
my_calculator = Calculator()
s = my_calculator.divisorSum(n)
print('I implemented: ' + type(my_calculator).__bases__[0].__name__)
print(s)