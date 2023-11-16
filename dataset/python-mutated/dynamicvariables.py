import threading
from contextlib import contextmanager

class DynamicVariable:

    def __init__(self, default):
        if False:
            for i in range(10):
                print('nop')
        self.default = default
        self.data = threading.local()

    @property
    def value(self):
        if False:
            print('Hello World!')
        return getattr(self.data, 'value', self.default)

    @value.setter
    def value(self, value):
        if False:
            print('Hello World!')
        self.data.value = value

    @contextmanager
    def with_value(self, value):
        if False:
            print('Hello World!')
        old_value = self.value
        try:
            self.data.value = value
            yield
        finally:
            self.data.value = old_value