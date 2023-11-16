"""
Test machar. Given recent changes to hardcode type data, we might want to get
rid of both MachAr and this test at some point.

"""
from numpy._core._machar import MachAr
import numpy._core.numerictypes as ntypes
from numpy import errstate, array

class TestMachAr:

    def _run_machar_highprec(self):
        if False:
            return 10
        try:
            hiprec = ntypes.float96
            MachAr(lambda v: array(v, hiprec))
        except AttributeError:
            'Skipping test: no ntypes.float96 available on this platform.'

    def test_underlow(self):
        if False:
            print('Hello World!')
        with errstate(all='raise'):
            try:
                self._run_machar_highprec()
            except FloatingPointError as e:
                msg = 'Caught %s exception, should not have been raised.' % e
                raise AssertionError(msg)