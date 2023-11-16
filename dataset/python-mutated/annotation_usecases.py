"""
Usecases with Python 3 function annotations.  This is a separate module
in order to avoid syntax errors with Python 2.
"""

class AnnotatedClass:
    """
    A class with annotated methods.
    """

    def __init__(self, v: int):
        if False:
            return 10
        self.x = v

    def add(self, v: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.x + v