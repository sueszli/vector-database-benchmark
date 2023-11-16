class Calculation:

    def _make_ints(self, *args):
        if False:
            for i in range(10):
                print('nop')
        try:
            return [int(arg) for arg in args]
        except ValueError:
            raise TypeError("Couldn't coerce arguments to integers: {}".format(*args))

    def add(self, a, b):
        if False:
            return 10
        (a, b) = self._make_ints(a, b)
        return a + b

    def subtract(self, a, b):
        if False:
            print('Hello World!')
        (a, b) = self._make_ints(a, b)
        return a - b

    def multiply(self, a, b):
        if False:
            return 10
        (a, b) = self._make_ints(a, b)
        return a * b

    def divide(self, a, b):
        if False:
            return 10
        (a, b) = self._make_ints(a, b)
        return a // b