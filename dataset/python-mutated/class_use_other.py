class A:

    def store(a, b):
        if False:
            for i in range(10):
                print('nop')
        a.value = b

class B:
    pass
b = B()
A.store(b, 1)
print(b.value)