class C:
    from typing import overload

    @overload
    def f(self, x: int, y: int) -> None:
        if False:
            while True:
                i = 10
        ...

    def f(self, x, y):
        if False:
            while True:
                i = 10
        pass