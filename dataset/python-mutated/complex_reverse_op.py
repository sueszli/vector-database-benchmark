class A:

    def __radd__(self, x):
        if False:
            print('Hello World!')
        print('__radd__')
        return 2
print(1j + A())