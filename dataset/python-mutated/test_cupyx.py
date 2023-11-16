import unittest
import cupyx
from cupy import testing

@testing.parameterize(*testing.product({'divide': [None]}))
class TestErrState(unittest.TestCase):

    def test_errstate(self):
        if False:
            print('Hello World!')
        orig = cupyx.geterr()
        with cupyx.errstate(divide=self.divide):
            state = cupyx.geterr()
            assert state.pop('divide') == self.divide
            orig.pop('divide')
            assert state == orig

    def test_seterr(self):
        if False:
            print('Hello World!')
        pass