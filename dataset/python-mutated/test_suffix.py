from trashcli.put.core.int_generator import IntGenerator
from trashcli.put.suffix import Suffix

class TestSuffix:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.suffix = Suffix(InlineFakeIntGen(lambda x, y: '%s,%s' % (x, y)))

    def test_first_attempt(self):
        if False:
            return 10
        assert self.suffix.suffix_for_index(0) == ''

    def test_second_attempt(self):
        if False:
            i = 10
            return i + 15
        assert self.suffix.suffix_for_index(1) == '_1'

    def test_hundredth_attempt(self):
        if False:
            return 10
        assert self.suffix.suffix_for_index(100) == '_0,65535'

class InlineFakeIntGen(IntGenerator):

    def __init__(self, func):
        if False:
            return 10
        self.func = func

    def new_int(self, a, b):
        if False:
            print('Hello World!')
        return self.func(a, b)