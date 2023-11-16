from collections import OrderedDict
from django.contrib.auth.hashers import BasePasswordHasher

class DummyHasher(BasePasswordHasher):
    """Dummy password hasher used only for unit tests purpose.

    Overwriting default Django password hasher significantly reduces the time
    of test execution.
    """
    algorithm = 'dummy'

    def encode(self, password, *args):
        if False:
            i = 10
            return i + 15
        assert password is not None
        return f'{self.algorithm}${password}'

    def verify(self, password, encoded):
        if False:
            i = 10
            return i + 15
        (algorithm, dummy_password) = encoded.split('$')
        assert algorithm == self.algorithm
        return password == dummy_password

    def safe_summary(self, encoded):
        if False:
            for i in range(10):
                print('nop')
        (algorithm, dummy_password) = encoded.split('$')
        return OrderedDict([algorithm, dummy_password])

    def harden_runtime(self, password, encoded):
        if False:
            while True:
                i = 10
        pass