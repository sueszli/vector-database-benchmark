class MyStopIteration(StopIteration):
    pass

class myiter:

    def __init__(self, i):
        if False:
            while True:
                i = 10
        self.i = i

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        if self.i == 0:
            raise StopIteration
        elif self.i == 1:
            raise StopIteration(1)
        elif self.i == 2:
            raise MyStopIteration
print(list(myiter(0)))
print(list(myiter(1)))
print(list(myiter(2)))