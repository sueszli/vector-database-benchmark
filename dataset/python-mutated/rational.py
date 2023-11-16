from cupy import _core

def _negative_gcd_error():
    if False:
        i = 10
        return i + 15
    raise TypeError('gcd cannot be computed with boolean arrays')

def _negative_lcm_error():
    if False:
        i = 10
        return i + 15
    raise TypeError('lcm cannot be computed with boolean arrays')
_gcd_preamble = '\ntemplate <typename T> inline __device__ T gcd(T in0, T in1) {\n  T r;\n  while (in1 != 0) {\n    r = in0 % in1;\n    in0 = in1;\n    in1 = r;\n  }\n  if (in0 < 0)\n    return -in0;\n  return in0;\n}\n'
gcd = _core.create_ufunc('cupy_gcd', (('??->?', _negative_gcd_error), 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L', 'qq->q', 'QQ->Q'), 'out0 = gcd(in0, in1)', preamble=_gcd_preamble, doc='Computes gcd of ``x1`` and ``x2`` elementwise.\n\n    .. seealso:: :data:`numpy.gcd`\n\n    ')
_lcm_preamble = _gcd_preamble + '\ntemplate <typename T> inline __device__ T lcm(T in0, T in1) {\n  T r = gcd(in0, in1);\n  if (r == 0)\n    return 0;\n  r = in0 / r * in1;\n  if (r < 0)\n    return -r;\n  return r;\n}\n'
lcm = _core.create_ufunc('cupy_lcm', (('??->?', _negative_lcm_error), 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L', 'qq->q', 'QQ->Q'), 'out0 = lcm(in0, in1)', preamble=_lcm_preamble, doc='Computes lcm of ``x1`` and ``x2`` elementwise.\n\n    .. seealso:: :data:`numpy.lcm`\n\n    ')