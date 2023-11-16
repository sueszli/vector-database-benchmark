class C1:

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

    def __str__(self):
        if False:
            return 10
        return 'str<C1 {}>'.format(self.value)

class C2:

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'repr<C2 {}>'.format(self.value)

class C3:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'str<C3 {}>'.format(self.value)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'repr<C3 {}>'.format(self.value)
c1 = C1(1)
print(c1)
c2 = C2(2)
print(c2)
s11 = str(c1)
print(s11)
s12 = repr(c1)
print('C1 object at' in s12)
s21 = str(c2)
print(s21)
s22 = repr(c2)
print(s22)
c3 = C3(1)
print(c3)