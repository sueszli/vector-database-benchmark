import itertools
import unittest
from numba.core.compiler import compile_isolated
from numba.core import types

def template(fromty, toty):
    if False:
        while True:
            i = 10

    def closure(self):
        if False:
            for i in range(10):
                print('nop')

        def cast(x):
            if False:
                return 10
            y = x
            return y
        cres = compile_isolated(cast, args=[fromty], return_type=toty)
        self.assertAlmostEqual(cres.entry_point(1), 1)
    return closure

class TestNumberConversion(unittest.TestCase):
    """
    Test all int/float numeric conversion to ensure we have all the external
    dependencies to perform these conversions.
    """

    @classmethod
    def automatic_populate(cls):
        if False:
            print('Hello World!')
        tys = types.integer_domain | types.real_domain
        for (fromty, toty) in itertools.permutations(tys, r=2):
            test_name = 'test_{fromty}_to_{toty}'.format(fromty=fromty, toty=toty)
            setattr(cls, test_name, template(fromty, toty))
TestNumberConversion.automatic_populate()
if __name__ == '__main__':
    unittest.main()