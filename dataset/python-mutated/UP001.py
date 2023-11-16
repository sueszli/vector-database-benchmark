class A:
    __metaclass__ = type

class B:
    __metaclass__ = type

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pass

class C(metaclass=type):
    pass