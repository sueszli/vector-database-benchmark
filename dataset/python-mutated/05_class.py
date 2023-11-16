n
import io

class BZ2File(io.BufferedIOBase):
    pass

class ABC(metaclass=BZ2File):
    pass

def test_customdescriptors_with_abstractmethod():
    if False:
        print('Hello World!')

    class Descriptor:

        def setter(self):
            if False:
                print('Hello World!')
            return Descriptor(self._fget)