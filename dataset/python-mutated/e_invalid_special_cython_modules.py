import cython.imports.libc as libc_import
import cython.cimports.labc as labc_cimport
from cython.imports import libc
from cython.cimport.libc import math
from cython.imports.libc import math
from cython.cimports.labc import math
import cython.paralel
import cython.parrallel
import cython.dataclass
import cython.floating
import cython.cfunc
from cython.cimports.libc import math
from cython.cimports.libc.math import ceil

def libc_math_ceil(x):
    if False:
        i = 10
        return i + 15
    '\n    >>> libc_math_ceil(1.5)\n    [2, 2]\n    '
    return [int(n) for n in [ceil(x), math.ceil(x)]]
_ERRORS = "\n6:7: 'cython.imports.libc' is not a valid cython.* module. Did you mean 'cython.cimports' ?\n7:7: 'labc.pxd' not found\n9:0: 'cython.imports' is not a valid cython.* module. Did you mean 'cython.cimports' ?\n10:0: 'cython.cimport.libc' is not a valid cython.* module. Did you mean 'cython.cimports' ?\n11:0: 'cython.imports.libc' is not a valid cython.* module. Did you mean 'cython.cimports' ?\n12:0: 'labc/math.pxd' not found\n14:7: 'cython.paralel' is not a valid cython.* module. Did you mean 'cython.parallel' ?\n15:7: 'cython.parrallel' is not a valid cython.* module. Did you mean 'cython.parallel' ?\n17:7: 'cython.dataclass' is not a valid cython.* module. Did you mean 'cython.dataclasses' ?\n18:7: 'cython.floating' is not a valid cython.* module. Instead, use 'import cython' and then 'cython.floating'.\n19:7: 'cython.cfunc' is not a valid cython.* module. Instead, use 'import cython' and then 'cython.cfunc'.\n"