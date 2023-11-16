class C:

    def a(self, default=[]):
        if False:
            while True:
                i = 10
        self.default = default

    def b(self, x):
        if False:
            return 10
        self.default.append(1)

    def some_other_function(self):
        if False:
            return 10
        x = 5
        return x

class D:

    def b(self, x):
        if False:
            i = 10
            return i + 15
        self.default.append(1)

    def a(self, default=[]):
        if False:
            for i in range(10):
                print('nop')
        self.default = default

    def some_other_function(self):
        if False:
            for i in range(10):
                print('nop')
        x = 5
        return x

class E:

    def a(self, default=[]):
        if False:
            i = 10
            return i + 15
        self.default = default

    def some_other_function(self):
        if False:
            i = 10
            return i + 15
        x = 5
        return x

    def b(self, x):
        if False:
            print('Hello World!')
        self.default.append(1)