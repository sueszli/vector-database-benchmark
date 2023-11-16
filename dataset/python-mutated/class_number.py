class C:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def __add__(self, rhs):
        if False:
            print('Hello World!')
        print(self.value, '+', rhs)

    def __sub__(self, rhs):
        if False:
            while True:
                i = 10
        print(self.value, '-', rhs)
c = C(0)
c + 1
c - 2