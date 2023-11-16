class E:

    def __repr__(self):
        if False:
            return 10
        return 'E'

    def __eq__(self, other):
        if False:
            print('Hello World!')
        print('E eq', other)
        return 123

class F:

    def __repr__(self):
        if False:
            return 10
        return 'F'

    def __ne__(self, other):
        if False:
            print('Hello World!')
        print('F ne', other)
        return -456
print(E() != F())
print(F() != E())
tests = (None, 0, 1, 'a')
for val in tests:
    print('==== testing', val)
    print(E() == val)
    print(val == E())
    print(E() != val)
    print(val != E())
    print(F() == val)
    print(val == F())
    print(F() != val)
    print(val != F())