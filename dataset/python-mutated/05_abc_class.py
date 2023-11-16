class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        if False:
            print('Hello World!')
        callable.__isabstractmethod__ = True
        super().__init__(callable)

class OldstyleClass:
    pass