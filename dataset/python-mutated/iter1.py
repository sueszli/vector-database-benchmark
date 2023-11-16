class NotIterable:
    pass
try:
    for i in NotIterable():
        pass
except TypeError:
    print('TypeError')

class NotIterable:

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self
try:
    print(all(NotIterable()))
except TypeError:
    print('TypeError')

class MyStopIteration(StopIteration):
    pass

class myiter:

    def __init__(self, i):
        if False:
            print('Hello World!')
        self.i = i

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        if self.i <= 0:
            raise StopIteration
        elif self.i == 10:
            raise StopIteration(42)
        elif self.i == 20:
            raise TypeError
        elif self.i == 30:
            print('raising MyStopIteration')
            raise MyStopIteration
        else:
            self.i -= 1
            return self.i
for i in myiter(5):
    print(i)
for i in myiter(12):
    print(i)
try:
    for i in myiter(22):
        print(i)
except TypeError:
    print('raised TypeError')
try:
    for i in myiter(5):
        print(i)
        raise StopIteration
except StopIteration:
    print('raised StopIteration')
for i in myiter(32):
    print(i)
print(tuple(myiter(5)))
print(tuple(myiter(12)))
print(tuple(myiter(32)))
try:
    tuple(myiter(22))
except TypeError:
    print('raised TypeError')