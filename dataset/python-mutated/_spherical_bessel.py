import cupy
from cupy import _core
spherical_bessel_preamble = '\n#include <cupy/math_constants.h>\n\n__device__ double spherical_yn_real(int n, double x) {\n    double s, s0, s1;\n\n    if (isnan(x))\n        return x;\n    if (x < 0) {\n        if (n % 2 == 0)\n            return -spherical_yn_real(n, -x);\n        else\n            return spherical_yn_real(n, -x);\n    }\n    if (isinf(x))\n        return 0;\n    if (x == 0)\n        return -CUDART_INF;\n\n    s0 = -cos(x) / x;\n    if (n == 0) {\n        return s0;\n    }\n    s1 = (s0 - sin(x)) / x;\n    for (int k = 2; k <= n; ++k) {\n        s = (2.0 * k - 1.0) * s1 / x - s0;\n        if (isinf(s)) {\n            return s;\n        }\n        s0 = s1;\n        s1 = s;\n    }\n\n    return s1;\n}\n\n__device__ double spherical_yn_d_real(int n, double x) {\n    double s, s0, s1;\n\n    if (isnan(x))\n        return x;\n    if (x < 0) {\n        if (n % 2 == 0)\n            return -spherical_yn_d_real(n, -x);\n        else\n            return spherical_yn_d_real(n, -x);\n    }\n    if (isinf(x))\n        return 0;\n    if (x == 0)\n        return CUDART_INF;\n\n    if (n == 1) {\n        return (sin(x) + cos(x) / x) / x;\n    }\n    s0 = -cos(x) / x;\n    s1 = (s0 - sin(x)) / x;\n    for (int k = 2; k <= n; ++k) {\n        s = (2.0 * k - 1.0) * s1 / x - s0;\n        if (isinf(s)) {\n            return s;\n        }\n        s0 = s1;\n        s1 = s;\n    }\n\n    return s0 - (n + 1.0) * s1 / x;\n}\n'
_spherical_yn_real = _core.create_ufunc('cupyx_scipy_spherical_yn_real', ('if->d', 'id->d'), 'out0 = out0_type(spherical_yn_real(in0, in1))', preamble=spherical_bessel_preamble)
_spherical_dyn_real = _core.create_ufunc('cupyx_scipy_spherical_dyn_real', ('if->d', 'id->d'), 'out0 = out0_type(spherical_yn_d_real(in0, in1));', preamble=spherical_bessel_preamble)

def spherical_yn(n, z, derivative=False):
    if False:
        for i in range(10):
            print('nop')
    'Spherical Bessel function of the second kind or its derivative.\n\n    Parameters\n    ----------\n    n : cupy.ndarray\n        Order of the Bessel function.\n    z : cupy.ndarray\n        Argument of the Bessel function.\n        Real-valued input.\n    derivative : bool, optional\n        If True, the value of the derivative (rather than the function\n        itself) is returned.\n\n    Returns\n    -------\n    yn : cupy.ndarray\n\n    See Also\n    -------\n    :func:`scipy.special.spherical_yn`\n\n    '
    if cupy.iscomplexobj(z):
        if derivative:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif derivative:
        return _spherical_dyn_real(n, z)
    else:
        return _spherical_yn_real(n, z)