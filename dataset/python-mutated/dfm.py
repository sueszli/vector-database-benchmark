"""
sympy.polys.matrices.dfm

Provides the :class:`DFM` class if ``GROUND_TYPES=flint'``. Otherwise, ``DFM``
is a placeholder class that raises NotImplementedError when instantiated.
"""
from sympy.external.gmpy import GROUND_TYPES
if GROUND_TYPES == 'flint':
    from ._dfm import DFM
else:

    class DFM_dummy:
        """
        Placeholder class for DFM when python-flint is not installed.
        """

        def __init__(*args, **kwargs):
            if False:
                print('Hello World!')
            raise NotImplementedError('DFM requires GROUND_TYPES=flint.')

        @classmethod
        def _supports_domain(cls, domain):
            if False:
                return 10
            return False

        @classmethod
        def _get_flint_func(cls, domain):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError('DFM requires GROUND_TYPES=flint.')
    DFM = DFM_dummy