try:
    reversed
except:
    print('SKIP')
    raise SystemExit
print(list(reversed([])))
print(list(reversed([1])))
print(list(reversed([1, 2, 3])))
print(list(reversed(())))
print(list(reversed((1, 2, 3))))
for c in reversed('ab'):
    print(c)
for b in reversed(b'1234'):
    print(b)
for i in reversed(range(3)):
    print(i)

class A:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return 3

    def __getitem__(self, pos):
        if False:
            return 10
        return pos + 1
for a in reversed(A()):
    print(a)

class B:

    def __reversed__(self):
        if False:
            while True:
                i = 10
        return [1, 2, 3]
print(reversed(B()))