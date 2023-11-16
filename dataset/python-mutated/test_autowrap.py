import os
import tempfile
import shutil
from io import StringIO
from sympy.core import symbols, Eq
from sympy.utilities.autowrap import autowrap, binary_function, CythonCodeWrapper, UfuncifyCodeWrapper, CodeWrapper
from sympy.utilities.codegen import CCodeGen, C99CodeGen, CodeGenArgumentListError, make_routine
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager

def get_string(dump_fn, routines, prefix='file', **kwargs):
    if False:
        print('Hello World!')
    'Wrapper for dump_fn. dump_fn writes its results to a stream object and\n       this wrapper returns the contents of that stream as a string. This\n       auxiliary function is used by many tests below.\n\n       The header and the empty lines are not generator to facilitate the\n       testing of the output.\n    '
    output = StringIO()
    dump_fn(routines, output, prefix, **kwargs)
    source = output.getvalue()
    output.close()
    return source

def test_cython_wrapper_scalar_function():
    if False:
        print('Hello World!')
    (x, y, z) = symbols('x,y,z')
    expr = (x + y) * z
    routine = make_routine('test', expr)
    code_gen = CythonCodeWrapper(CCodeGen())
    source = get_string(code_gen.dump_pyx, [routine])
    expected = "cdef extern from 'file.h':\n    double test(double x, double y, double z)\n\ndef test_c(double x, double y, double z):\n\n    return test(x, y, z)"
    assert source == expected

def test_cython_wrapper_outarg():
    if False:
        print('Hello World!')
    from sympy.core.relational import Equality
    (x, y, z) = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())
    routine = make_routine('test', Equality(z, x + y))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = "cdef extern from 'file.h':\n    void test(double x, double y, double *z)\n\ndef test_c(double x, double y):\n\n    cdef double z = 0\n    test(x, y, &z)\n    return z"
    assert source == expected

def test_cython_wrapper_inoutarg():
    if False:
        i = 10
        return i + 15
    from sympy.core.relational import Equality
    (x, y, z) = symbols('x,y,z')
    code_gen = CythonCodeWrapper(C99CodeGen())
    routine = make_routine('test', Equality(z, x + y + z))
    source = get_string(code_gen.dump_pyx, [routine])
    expected = "cdef extern from 'file.h':\n    void test(double x, double y, double *z)\n\ndef test_c(double x, double y, double z):\n\n    test(x, y, &z)\n    return z"
    assert source == expected

def test_cython_wrapper_compile_flags():
    if False:
        while True:
            i = 10
    from sympy.core.relational import Equality
    (x, y, z) = symbols('x,y,z')
    routine = make_routine('test', Equality(z, x + y))
    code_gen = CythonCodeWrapper(CCodeGen())
    expected = "from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {'compiler_directives': {'language_level': '3'}}\n\next_mods = [Extension(\n    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],\n    include_dirs=[],\n    library_dirs=[],\n    libraries=[],\n    extra_compile_args=['-std=c99'],\n    extra_link_args=[]\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n" % {'num': CodeWrapper._module_counter}
    temp_dir = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    setup_file_path = os.path.join(temp_dir, 'setup.py')
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected
    code_gen = CythonCodeWrapper(CCodeGen(), include_dirs=['/usr/local/include', '/opt/booger/include'], library_dirs=['/user/local/lib'], libraries=['thelib', 'nilib'], extra_compile_args=['-slow-math'], extra_link_args=['-lswamp', '-ltrident'], cythonize_options={'compiler_directives': {'boundscheck': False}})
    expected = "from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {'compiler_directives': {'boundscheck': False}}\n\next_mods = [Extension(\n    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],\n    include_dirs=['/usr/local/include', '/opt/booger/include'],\n    library_dirs=['/user/local/lib'],\n    libraries=['thelib', 'nilib'],\n    extra_compile_args=['-slow-math', '-std=c99'],\n    extra_link_args=['-lswamp', '-ltrident']\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n" % {'num': CodeWrapper._module_counter}
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected
    expected = "from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {'compiler_directives': {'boundscheck': False}}\nimport numpy as np\n\next_mods = [Extension(\n    'wrapper_module_%(num)s', ['wrapper_module_%(num)s.pyx', 'wrapped_code_%(num)s.c'],\n    include_dirs=['/usr/local/include', '/opt/booger/include', np.get_include()],\n    library_dirs=['/user/local/lib'],\n    libraries=['thelib', 'nilib'],\n    extra_compile_args=['-slow-math', '-std=c99'],\n    extra_link_args=['-lswamp', '-ltrident']\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n" % {'num': CodeWrapper._module_counter}
    code_gen._need_numpy = True
    code_gen._prepare_files(routine, build_dir=temp_dir)
    with open(setup_file_path) as f:
        setup_text = f.read()
    assert setup_text == expected
    TmpFileManager.cleanup()

def test_cython_wrapper_unique_dummyvars():
    if False:
        print('Hello World!')
    from sympy.core.relational import Equality
    from sympy.core.symbol import Dummy
    (x, y, z) = (Dummy('x'), Dummy('y'), Dummy('z'))
    (x_id, y_id, z_id) = [str(d.dummy_index) for d in [x, y, z]]
    expr = Equality(z, x + y)
    routine = make_routine('test', expr)
    code_gen = CythonCodeWrapper(CCodeGen())
    source = get_string(code_gen.dump_pyx, [routine])
    expected_template = "cdef extern from 'file.h':\n    void test(double x_{x_id}, double y_{y_id}, double *z_{z_id})\n\ndef test_c(double x_{x_id}, double y_{y_id}):\n\n    cdef double z_{z_id} = 0\n    test(x_{x_id}, y_{y_id}, &z_{z_id})\n    return z_{z_id}"
    expected = expected_template.format(x_id=x_id, y_id=y_id, z_id=z_id)
    assert source == expected

def test_autowrap_dummy():
    if False:
        for i in range(10):
            print('nop')
    (x, y, z) = symbols('x y z')
    f = autowrap(x + y, backend='dummy')
    assert f() == str(x + y)
    assert f.args == 'x, y'
    assert f.returns == 'nameless'
    f = autowrap(Eq(z, x + y), backend='dummy')
    assert f() == str(x + y)
    assert f.args == 'x, y'
    assert f.returns == 'z'
    f = autowrap(Eq(z, x + y + z), backend='dummy')
    assert f() == str(x + y + z)
    assert f.args == 'x, y, z'
    assert f.returns == 'z'

def test_autowrap_args():
    if False:
        while True:
            i = 10
    (x, y, z) = symbols('x y z')
    raises(CodeGenArgumentListError, lambda : autowrap(Eq(z, x + y), backend='dummy', args=[x]))
    f = autowrap(Eq(z, x + y), backend='dummy', args=[y, x])
    assert f() == str(x + y)
    assert f.args == 'y, x'
    assert f.returns == 'z'
    raises(CodeGenArgumentListError, lambda : autowrap(Eq(z, x + y + z), backend='dummy', args=[x, y]))
    f = autowrap(Eq(z, x + y + z), backend='dummy', args=[y, x, z])
    assert f() == str(x + y + z)
    assert f.args == 'y, x, z'
    assert f.returns == 'z'
    f = autowrap(Eq(z, x + y + z), backend='dummy', args=(y, x, z))
    assert f() == str(x + y + z)
    assert f.args == 'y, x, z'
    assert f.returns == 'z'

def test_autowrap_store_files():
    if False:
        print('Hello World!')
    (x, y) = symbols('x y')
    tmp = tempfile.mkdtemp()
    TmpFileManager.tmp_folder(tmp)
    f = autowrap(x + y, backend='dummy', tempdir=tmp)
    assert f() == str(x + y)
    assert os.access(tmp, os.F_OK)
    TmpFileManager.cleanup()

def test_autowrap_store_files_issue_gh12939():
    if False:
        print('Hello World!')
    (x, y) = symbols('x y')
    tmp = './tmp'
    saved_cwd = os.getcwd()
    temp_cwd = tempfile.mkdtemp()
    try:
        os.chdir(temp_cwd)
        f = autowrap(x + y, backend='dummy', tempdir=tmp)
        assert f() == str(x + y)
        assert os.access(tmp, os.F_OK)
    finally:
        os.chdir(saved_cwd)
        shutil.rmtree(temp_cwd)

def test_binary_function():
    if False:
        return 10
    (x, y) = symbols('x y')
    f = binary_function('f', x + y, backend='dummy')
    assert f._imp_() == str(x + y)

def test_ufuncify_source():
    if False:
        for i in range(10):
            print('nop')
    (x, y, z) = symbols('x,y,z')
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen('ufuncify'))
    routine = make_routine('test', x + y + z)
    source = get_string(code_wrapper.dump_c, [routine])
    expected = '#include "Python.h"\n#include "math.h"\n#include "numpy/ndarraytypes.h"\n#include "numpy/ufuncobject.h"\n#include "numpy/halffloat.h"\n#include "file.h"\n\nstatic PyMethodDef wrapper_module_%(num)sMethods[] = {\n        {NULL, NULL, 0, NULL}\n};\n\nstatic void test_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)\n{\n    npy_intp i;\n    npy_intp n = dimensions[0];\n    char *in0 = args[0];\n    char *in1 = args[1];\n    char *in2 = args[2];\n    char *out0 = args[3];\n    npy_intp in0_step = steps[0];\n    npy_intp in1_step = steps[1];\n    npy_intp in2_step = steps[2];\n    npy_intp out0_step = steps[3];\n    for (i = 0; i < n; i++) {\n        *((double *)out0) = test(*(double *)in0, *(double *)in1, *(double *)in2);\n        in0 += in0_step;\n        in1 += in1_step;\n        in2 += in2_step;\n        out0 += out0_step;\n    }\n}\nPyUFuncGenericFunction test_funcs[1] = {&test_ufunc};\nstatic char test_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};\nstatic void *test_data[1] = {NULL};\n\n#if PY_VERSION_HEX >= 0x03000000\nstatic struct PyModuleDef moduledef = {\n    PyModuleDef_HEAD_INIT,\n    "wrapper_module_%(num)s",\n    NULL,\n    -1,\n    wrapper_module_%(num)sMethods,\n    NULL,\n    NULL,\n    NULL,\n    NULL\n};\n\nPyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)\n{\n    PyObject *m, *d;\n    PyObject *ufunc0;\n    m = PyModule_Create(&moduledef);\n    if (!m) {\n        return NULL;\n    }\n    import_array();\n    import_umath();\n    d = PyModule_GetDict(m);\n    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,\n            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);\n    PyDict_SetItemString(d, "test", ufunc0);\n    Py_DECREF(ufunc0);\n    return m;\n}\n#else\nPyMODINIT_FUNC initwrapper_module_%(num)s(void)\n{\n    PyObject *m, *d;\n    PyObject *ufunc0;\n    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);\n    if (m == NULL) {\n        return;\n    }\n    import_array();\n    import_umath();\n    d = PyModule_GetDict(m);\n    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,\n            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);\n    PyDict_SetItemString(d, "test", ufunc0);\n    Py_DECREF(ufunc0);\n}\n#endif' % {'num': CodeWrapper._module_counter}
    assert source == expected

def test_ufuncify_source_multioutput():
    if False:
        return 10
    (x, y, z) = symbols('x,y,z')
    var_symbols = (x, y, z)
    expr = x + y ** 3 + 10 * z ** 2
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen('ufuncify'))
    routines = [make_routine('func{}'.format(i), expr.diff(var_symbols[i]), var_symbols) for i in range(len(var_symbols))]
    source = get_string(code_wrapper.dump_c, routines, funcname='multitest')
    expected = '#include "Python.h"\n#include "math.h"\n#include "numpy/ndarraytypes.h"\n#include "numpy/ufuncobject.h"\n#include "numpy/halffloat.h"\n#include "file.h"\n\nstatic PyMethodDef wrapper_module_%(num)sMethods[] = {\n        {NULL, NULL, 0, NULL}\n};\n\nstatic void multitest_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)\n{\n    npy_intp i;\n    npy_intp n = dimensions[0];\n    char *in0 = args[0];\n    char *in1 = args[1];\n    char *in2 = args[2];\n    char *out0 = args[3];\n    char *out1 = args[4];\n    char *out2 = args[5];\n    npy_intp in0_step = steps[0];\n    npy_intp in1_step = steps[1];\n    npy_intp in2_step = steps[2];\n    npy_intp out0_step = steps[3];\n    npy_intp out1_step = steps[4];\n    npy_intp out2_step = steps[5];\n    for (i = 0; i < n; i++) {\n        *((double *)out0) = func0(*(double *)in0, *(double *)in1, *(double *)in2);\n        *((double *)out1) = func1(*(double *)in0, *(double *)in1, *(double *)in2);\n        *((double *)out2) = func2(*(double *)in0, *(double *)in1, *(double *)in2);\n        in0 += in0_step;\n        in1 += in1_step;\n        in2 += in2_step;\n        out0 += out0_step;\n        out1 += out1_step;\n        out2 += out2_step;\n    }\n}\nPyUFuncGenericFunction multitest_funcs[1] = {&multitest_ufunc};\nstatic char multitest_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};\nstatic void *multitest_data[1] = {NULL};\n\n#if PY_VERSION_HEX >= 0x03000000\nstatic struct PyModuleDef moduledef = {\n    PyModuleDef_HEAD_INIT,\n    "wrapper_module_%(num)s",\n    NULL,\n    -1,\n    wrapper_module_%(num)sMethods,\n    NULL,\n    NULL,\n    NULL,\n    NULL\n};\n\nPyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)\n{\n    PyObject *m, *d;\n    PyObject *ufunc0;\n    m = PyModule_Create(&moduledef);\n    if (!m) {\n        return NULL;\n    }\n    import_array();\n    import_umath();\n    d = PyModule_GetDict(m);\n    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,\n            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);\n    PyDict_SetItemString(d, "multitest", ufunc0);\n    Py_DECREF(ufunc0);\n    return m;\n}\n#else\nPyMODINIT_FUNC initwrapper_module_%(num)s(void)\n{\n    PyObject *m, *d;\n    PyObject *ufunc0;\n    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);\n    if (m == NULL) {\n        return;\n    }\n    import_array();\n    import_umath();\n    d = PyModule_GetDict(m);\n    ufunc0 = PyUFunc_FromFuncAndData(multitest_funcs, multitest_data, multitest_types, 1, 3, 3,\n            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);\n    PyDict_SetItemString(d, "multitest", ufunc0);\n    Py_DECREF(ufunc0);\n}\n#endif' % {'num': CodeWrapper._module_counter}
    assert source == expected