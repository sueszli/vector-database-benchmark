"""Tests for moved modules and their redirection from old path
"""
from numba.tests.support import TestCase

class TestMovedModule(TestCase):
    """Testing moved modules in Q1 2020 but were decided to kept as public API
    """

    def tests_numba_types(self):
        if False:
            for i in range(10):
                print('nop')
        import numba.types
        import numba.core.types as types
        self.assertIsNot(numba.types, types)
        self.assertIs(numba.types.intp, types.intp)
        self.assertIs(numba.types.float64, types.float64)
        self.assertIs(numba.types.Array, types.Array)
        import numba.types.misc
        self.assertIs(types.misc, numba.types.misc)
        self.assertIs(types.misc.Optional, numba.types.misc.Optional)
        self.assertIs(types.StringLiteral, numba.types.misc.StringLiteral)
        from numba.types import containers
        self.assertIs(types.containers, containers)
        self.assertIs(types.containers.Sequence, containers.Sequence)
        from numba.types.containers import Sequence
        self.assertIs(Sequence, containers.Sequence)