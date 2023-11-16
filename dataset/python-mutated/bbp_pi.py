"""
This implementation is a heavily modified fixed point implementation of
BBP_formula for calculating the nth position of pi. The original hosted
at: https://web.archive.org/web/20151116045029/http://en.literateprograms.org/Pi_with_the_BBP_formula_(Python)

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sub-license, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications:

1.Once the nth digit and desired number of digits is selected, the
number of digits of working precision is calculated to ensure that
the hexadecimal digits returned are accurate. This is calculated as

    int(math.log(start + prec)/math.log(16) + prec + 3)
    ---------------------------------------   --------
                      /                          /
    number of hex digits           additional digits

This was checked by the following code which completed without
errors (and dig are the digits included in the test_bbp.py file):

    for i in range(0,1000):
     for j in range(1,1000):
      a, b = pi_hex_digits(i, j), dig[i:i+j]
      if a != b:
        print('%s
%s'%(a,b))

Deceasing the additional digits by 1 generated errors, so '3' is
the smallest additional precision needed to calculate the above
loop without errors. The following trailing 10 digits were also
checked to be accurate (and the times were slightly faster with
some of the constant modifications that were made):

    >> from time import time
    >> t=time();pi_hex_digits(10**2-10 + 1, 10), time()-t
    ('e90c6cc0ac', 0.0)
    >> t=time();pi_hex_digits(10**4-10 + 1, 10), time()-t
    ('26aab49ec6', 0.17100000381469727)
    >> t=time();pi_hex_digits(10**5-10 + 1, 10), time()-t
    ('a22673c1a5', 4.7109999656677246)
    >> t=time();pi_hex_digits(10**6-10 + 1, 10), time()-t
    ('9ffd342362', 59.985999822616577)
    >> t=time();pi_hex_digits(10**7-10 + 1, 10), time()-t
    ('c1a42e06a1', 689.51800012588501)

2. The while loop to evaluate whether the series has converged quits
when the addition amount `dt` has dropped to zero.

3. the formatting string to convert the decimal to hexadecimal is
calculated for the given precision.

4. pi_hex_digits(n) changed to have coefficient to the formula in an
array (perhaps just a matter of preference).

"""
from sympy.utilities.misc import as_int

def _series(j, n, prec=14):
    if False:
        return 10
    s = 0
    D = _dn(n, prec)
    D4 = 4 * D
    d = j
    for k in range(n + 1):
        s += (pow(16, n - k, d) << D4) // d
        d += 8
    t = 0
    k = n + 1
    e = D4 - 4
    d = 8 * k + j
    while True:
        dt = (1 << e) // d
        if not dt:
            break
        t += dt
        e -= 4
        d += 8
    total = s + t
    return total

def pi_hex_digits(n, prec=14):
    if False:
        for i in range(10):
            print('nop')
    "Returns a string containing ``prec`` (default 14) digits\n    starting at the nth digit of pi in hex. Counting of digits\n    starts at 0 and the decimal is not counted, so for n = 0 the\n    returned value starts with 3; n = 1 corresponds to the first\n    digit past the decimal point (which in hex is 2).\n\n    Parameters\n    ==========\n\n    n : non-negative integer\n    prec : non-negative integer. default = 14\n\n    Returns\n    =======\n\n    str : Returns a string containing ``prec`` digits\n          starting at the nth digit of pi in hex.\n          If ``prec`` = 0, returns empty string.\n\n    Raises\n    ======\n\n    ValueError\n        If ``n`` < 0 or ``prec`` < 0.\n        Or ``n`` or ``prec`` is not an integer.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.bbp_pi import pi_hex_digits\n    >>> pi_hex_digits(0)\n    '3243f6a8885a30'\n    >>> pi_hex_digits(0, 3)\n    '324'\n\n    These are consistent with the following results\n\n    >>> import math\n    >>> hex(int(math.pi * 2**((14-1)*4)))\n    '0x3243f6a8885a30'\n    >>> hex(int(math.pi * 2**((3-1)*4)))\n    '0x324'\n\n    References\n    ==========\n\n    .. [1] http://www.numberworld.org/digits/Pi/\n    "
    (n, prec) = (as_int(n), as_int(prec))
    if n < 0:
        raise ValueError('n cannot be negative')
    if prec < 0:
        raise ValueError('prec cannot be negative')
    if prec == 0:
        return ''
    n -= 1
    a = [4, 2, 1, 1]
    j = [1, 4, 5, 6]
    D = _dn(n, prec)
    x = +(a[0] * _series(j[0], n, prec) - a[1] * _series(j[1], n, prec) - a[2] * _series(j[2], n, prec) - a[3] * _series(j[3], n, prec)) & 16 ** D - 1
    s = ('%0' + '%ix' % prec) % (x // 16 ** (D - prec))
    return s

def _dn(n, prec):
    if False:
        i = 10
        return i + 15
    n += 1
    return ((n + prec).bit_length() - 1) // 4 + prec + 3