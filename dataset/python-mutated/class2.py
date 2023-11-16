class C1:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = 1
c1 = C1()
print(type(c1) == C1)
print(c1.x)

class C2:

    def __init__(self, x):
        if False:
            return 10
        self.x = x
c2 = C2(4)
print(type(c2) == C2)
print(c2.x)

class C3:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        return 10
try:
    C3()
except TypeError:
    print('TypeError')