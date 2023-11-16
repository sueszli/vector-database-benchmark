"""Test: inner class annotation."""

class RandomClass:

    def random_func(self) -> 'InnerClass':
        if False:
            while True:
                i = 10
        pass

class OuterClass:

    class InnerClass:
        pass

    def failing_func(self) -> 'InnerClass':
        if False:
            return 10
        return self.InnerClass()