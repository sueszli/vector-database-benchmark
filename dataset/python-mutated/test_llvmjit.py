from sympy.external import import_module
from sympy.testing.pytest import raises
import ctypes
if import_module('llvmlite'):
    import sympy.printing.llvmjitcode as g
else:
    disabled = True
import sympy
from sympy.abc import a, b, n

def isclose(a, b):
    if False:
        return 10
    rtol = 1e-05
    atol = 1e-08
    return abs(a - b) <= atol + rtol * abs(b)

def test_simple_expr():
    if False:
        while True:
            i = 10
    e = a + 1.0
    f = g.llvm_callable([a], e)
    res = float(e.subs({a: 4.0}).evalf())
    jit_res = f(4.0)
    assert isclose(jit_res, res)

def test_two_arg():
    if False:
        for i in range(10):
            print('nop')
    e = 4.0 * a + b + 3.0
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 4.0, b: 3.0}).evalf())
    jit_res = f(4.0, 3.0)
    assert isclose(jit_res, res)

def test_func():
    if False:
        while True:
            i = 10
    e = 4.0 * sympy.exp(-a)
    f = g.llvm_callable([a], e)
    res = float(e.subs({a: 1.5}).evalf())
    jit_res = f(1.5)
    assert isclose(jit_res, res)

def test_two_func():
    if False:
        print('Hello World!')
    e = 4.0 * sympy.exp(-a) + sympy.exp(b)
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)
    assert isclose(jit_res, res)

def test_two_sqrt():
    if False:
        i = 10
        return i + 15
    e = 4.0 * sympy.sqrt(a) + sympy.sqrt(b)
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)
    assert isclose(jit_res, res)

def test_two_pow():
    if False:
        i = 10
        return i + 15
    e = a ** 1.5 + b ** 7
    f = g.llvm_callable([a, b], e)
    res = float(e.subs({a: 1.5, b: 2.0}).evalf())
    jit_res = f(1.5, 2.0)
    assert isclose(jit_res, res)

def test_callback():
    if False:
        while True:
            i = 10
    e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    inp = {a: 2.2}
    array = array_type(inp[a])
    jit_res = f(m, array)
    res = float(e.subs(inp).evalf())
    assert isclose(jit_res, res)

def test_callback_cubature():
    if False:
        i = 10
        return i + 15
    e = a + 1.2
    f = g.llvm_callable([a], e, callback_type='cubature')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    inp = {a: 2.2}
    array = array_type(inp[a])
    out_array = array_type(0.0)
    jit_ret = f(m, array, None, m, out_array)
    assert jit_ret == 0
    res = float(e.subs(inp).evalf())
    assert isclose(out_array[0], res)

def test_callback_two():
    if False:
        print('Hello World!')
    e = 3 * a * b
    f = g.llvm_callable([a, b], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(2)
    array_type = ctypes.c_double * 2
    inp = {a: 0.2, b: 1.7}
    array = array_type(inp[a], inp[b])
    jit_res = f(m, array)
    res = float(e.subs(inp).evalf())
    assert isclose(jit_res, res)

def test_callback_alt_two():
    if False:
        while True:
            i = 10
    d = sympy.IndexedBase('d')
    e = 3 * d[0] * d[1]
    f = g.llvm_callable([n, d], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(2)
    array_type = ctypes.c_double * 2
    inp = {d[0]: 0.2, d[1]: 1.7}
    array = array_type(inp[d[0]], inp[d[1]])
    jit_res = f(m, array)
    res = float(e.subs(inp).evalf())
    assert isclose(jit_res, res)

def test_multiple_statements():
    if False:
        return 10
    e = [[(b, 4.0 * a)], [b + 5]]
    f = g.llvm_callable([a], e)
    b_val = e[0][0][1].subs({a: 1.5})
    res = float(e[1][0].subs({b: b_val}).evalf())
    jit_res = f(1.5)
    assert isclose(jit_res, res)
    f_callback = g.llvm_callable([a], e, callback_type='scipy.integrate.test')
    m = ctypes.c_int(1)
    array_type = ctypes.c_double * 1
    array = array_type(1.5)
    jit_callback_res = f_callback(m, array)
    assert isclose(jit_callback_res, res)

def test_cse():
    if False:
        print('Hello World!')
    e = a * a + b * b + sympy.exp(-a * a - b * b)
    e2 = sympy.cse(e)
    f = g.llvm_callable([a, b], e2)
    res = float(e.subs({a: 2.3, b: 0.1}).evalf())
    jit_res = f(2.3, 0.1)
    assert isclose(jit_res, res)

def eval_cse(e, sub_dict):
    if False:
        print('Hello World!')
    tmp_dict = {}
    for (tmp_name, tmp_expr) in e[0]:
        e2 = tmp_expr.subs(sub_dict)
        e3 = e2.subs(tmp_dict)
        tmp_dict[tmp_name] = e3
    return [e.subs(sub_dict).subs(tmp_dict) for e in e[1]]

def test_cse_multiple():
    if False:
        return 10
    e1 = a * a
    e2 = a * a + b * b
    e3 = sympy.cse([e1, e2])
    raises(NotImplementedError, lambda : g.llvm_callable([a, b], e3, callback_type='scipy.integrate'))
    f = g.llvm_callable([a, b], e3)
    jit_res = f(0.1, 1.5)
    assert len(jit_res) == 2
    res = eval_cse(e3, {a: 0.1, b: 1.5})
    assert isclose(res[0], jit_res[0])
    assert isclose(res[1], jit_res[1])

def test_callback_cubature_multiple():
    if False:
        return 10
    e1 = a * a
    e2 = a * a + b * b
    e3 = sympy.cse([e1, e2, 4 * e2])
    f = g.llvm_callable([a, b], e3, callback_type='cubature')
    ndim = 2
    outdim = 3
    m = ctypes.c_int(ndim)
    fdim = ctypes.c_int(outdim)
    array_type = ctypes.c_double * ndim
    out_array_type = ctypes.c_double * outdim
    inp = {a: 0.2, b: 1.5}
    array = array_type(inp[a], inp[b])
    out_array = out_array_type()
    jit_ret = f(m, array, None, fdim, out_array)
    assert jit_ret == 0
    res = eval_cse(e3, inp)
    assert isclose(out_array[0], res[0])
    assert isclose(out_array[1], res[1])
    assert isclose(out_array[2], res[2])

def test_symbol_not_found():
    if False:
        for i in range(10):
            print('nop')
    e = a * a + b
    raises(LookupError, lambda : g.llvm_callable([a], e))

def test_bad_callback():
    if False:
        i = 10
        return i + 15
    e = a
    raises(ValueError, lambda : g.llvm_callable([a], e, callback_type='bad_callback'))