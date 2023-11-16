class A:

    def __bool__(self):
        if False:
            while True:
                i = 10
        print('__bool__')
        return True

    def __len__(self):
        if False:
            while True:
                i = 10
        print('__len__')
        return 1

class B:

    def __len__(self):
        if False:
            i = 10
            return i + 15
        print('__len__')
        return 0
print(bool(A()))
print(len(A()))
print(bool(B()))
print(len(B()))