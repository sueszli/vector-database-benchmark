class A:

    def __add__(self, x):
        if False:
            i = 10
            return i + 15
        print('__add__')
        return 1
print(A() + 1j)