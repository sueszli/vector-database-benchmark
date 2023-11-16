import cython

@cython.annotation_typing(False)
def function_without_typing(a: int, b: int) -> int:
    if False:
        return 10
    'Cython is ignoring annotations in this function'
    c: int = a + b
    return c * a

@cython.annotation_typing(False)
@cython.cclass
class NotAnnotatedClass:
    """Cython is ignoring annotatons in this class except annotated_method"""
    d: dict

    def __init__(self, dictionary: dict):
        if False:
            while True:
                i = 10
        self.d = dictionary

    @cython.annotation_typing(True)
    def annotated_method(self, key: str, a: cython.int, b: cython.int):
        if False:
            while True:
                i = 10
        prefixed_key: str = 'prefix_' + key
        self.d[prefixed_key] = a + b

def annotated_function(a: cython.int, b: cython.int):
    if False:
        return 10
    s: cython.int = a + b
    with cython.annotation_typing(False):
        c: list = []
    c.append(a)
    c.append(b)
    c.append(s)
    return c