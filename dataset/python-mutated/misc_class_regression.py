class A:

    def method1(self, args):
        if False:
            i = 10
            return i + 15
        pass

class A2:

    def method2(self, args):
        if False:
            for i in range(10):
                print('nop')
        pass

class B:

    def method1(self, args):
        if False:
            print('Hello World!')
        print('hello there')

class C(A, B):

    def __init__():
        if False:
            i = 10
            return i + 15
        print('initialized')