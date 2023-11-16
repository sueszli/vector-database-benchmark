class C1:

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return self.value
c1 = C1(b'class 1')
print(bytes(c1))