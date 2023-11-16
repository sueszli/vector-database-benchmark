class A:

    def __eq__(self, other):
        if False:
            return 10
        print('A __eq__ called')
        return True

class B:

    def __ne__(self, other):
        if False:
            return 10
        print('B __ne__ called')
        return True

class C:

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        print('C __eq__ called')
        return False

class D:

    def __ne__(self, other):
        if False:
            print('Hello World!')
        print('D __ne__ called')
        return False
a = A()
b = B()
c = C()
d = D()

def test(s):
    if False:
        i = 10
        return i + 15
    print(s)
    print(eval(s))
for x in 'abcd':
    for y in 'abcd':
        test('{} == {}'.format(x, y))
        test('{} != {}'.format(x, y))