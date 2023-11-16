class CircularReferenceException(Exception):
    """
    Raised when a circular reference is detected in a manifest.
    """

    def __init__(self, reference):
        if False:
            while True:
                i = 10
        super().__init__(f'Circular reference found: {reference}')

class UndefinedReferenceException(Exception):
    """
    Raised when refering to an undefined reference.
    """

    def __init__(self, path, reference):
        if False:
            i = 10
            return i + 15
        super().__init__(f'Undefined reference {reference} from {path}')