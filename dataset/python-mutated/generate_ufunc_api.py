import os
import argparse
import genapi
from genapi import TypeApi, FunctionApi
import numpy_api
h_template = '\n#ifdef _UMATHMODULE\n\nextern NPY_NO_EXPORT PyTypeObject PyUFunc_Type;\n\n%s\n\n#else\n\n#if defined(PY_UFUNC_UNIQUE_SYMBOL)\n#define PyUFunc_API PY_UFUNC_UNIQUE_SYMBOL\n#endif\n\n#if defined(NO_IMPORT) || defined(NO_IMPORT_UFUNC)\nextern void **PyUFunc_API;\n#else\n#if defined(PY_UFUNC_UNIQUE_SYMBOL)\nvoid **PyUFunc_API;\n#else\nstatic void **PyUFunc_API=NULL;\n#endif\n#endif\n\n%s\n\nstatic inline int\n_import_umath(void)\n{\n  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");\n  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {\n    PyErr_Clear();\n    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");\n  }\n\n  if (numpy == NULL) {\n      PyErr_SetString(PyExc_ImportError,\n                      "_multiarray_umath failed to import");\n      return -1;\n  }\n\n  PyObject *c_api = PyObject_GetAttrString(numpy, "_UFUNC_API");\n  Py_DECREF(numpy);\n  if (c_api == NULL) {\n      PyErr_SetString(PyExc_AttributeError, "_UFUNC_API not found");\n      return -1;\n  }\n\n  if (!PyCapsule_CheckExact(c_api)) {\n      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is not PyCapsule object");\n      Py_DECREF(c_api);\n      return -1;\n  }\n  PyUFunc_API = (void **)PyCapsule_GetPointer(c_api, NULL);\n  Py_DECREF(c_api);\n  if (PyUFunc_API == NULL) {\n      PyErr_SetString(PyExc_RuntimeError, "_UFUNC_API is NULL pointer");\n      return -1;\n  }\n  return 0;\n}\n\n#define import_umath() \\\n    do {\\\n        UFUNC_NOFPE\\\n        if (_import_umath() < 0) {\\\n            PyErr_Print();\\\n            PyErr_SetString(PyExc_ImportError,\\\n                    "numpy._core.umath failed to import");\\\n            return NULL;\\\n        }\\\n    } while(0)\n\n#define import_umath1(ret) \\\n    do {\\\n        UFUNC_NOFPE\\\n        if (_import_umath() < 0) {\\\n            PyErr_Print();\\\n            PyErr_SetString(PyExc_ImportError,\\\n                    "numpy._core.umath failed to import");\\\n            return ret;\\\n        }\\\n    } while(0)\n\n#define import_umath2(ret, msg) \\\n    do {\\\n        UFUNC_NOFPE\\\n        if (_import_umath() < 0) {\\\n            PyErr_Print();\\\n            PyErr_SetString(PyExc_ImportError, msg);\\\n            return ret;\\\n        }\\\n    } while(0)\n\n#define import_ufunc() \\\n    do {\\\n        UFUNC_NOFPE\\\n        if (_import_umath() < 0) {\\\n            PyErr_Print();\\\n            PyErr_SetString(PyExc_ImportError,\\\n                    "numpy._core.umath failed to import");\\\n        }\\\n    } while(0)\n\n#endif\n'
c_template = '\n/* These pointers will be stored in the C-object for use in other\n    extension modules\n*/\n\nvoid *PyUFunc_API[] = {\n%s\n};\n'

def generate_api(output_dir, force=False):
    if False:
        for i in range(10):
            print('nop')
    basename = 'ufunc_api'
    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    targets = (h_file, c_file)
    sources = ['ufunc_api_order.txt']
    do_generate_api(targets, sources)
    return targets

def do_generate_api(targets, sources):
    if False:
        i = 10
        return i + 15
    header_file = targets[0]
    c_file = targets[1]
    ufunc_api_index = genapi.merge_api_dicts((numpy_api.ufunc_funcs_api, numpy_api.ufunc_types_api))
    genapi.check_api_dict(ufunc_api_index)
    ufunc_api_list = genapi.get_api_functions('UFUNC_API', numpy_api.ufunc_funcs_api)
    ufunc_api_dict = {}
    api_name = 'PyUFunc_API'
    for f in ufunc_api_list:
        name = f.name
        index = ufunc_api_index[name][0]
        annotations = ufunc_api_index[name][1:]
        ufunc_api_dict[name] = FunctionApi(f.name, index, annotations, f.return_type, f.args, api_name)
    for (name, val) in numpy_api.ufunc_types_api.items():
        index = val[0]
        ufunc_api_dict[name] = TypeApi(name, index, 'PyTypeObject', api_name)
    module_list = []
    extension_list = []
    init_list = []
    for (name, index) in genapi.order_dict(ufunc_api_index):
        api_item = ufunc_api_dict[name]
        while len(init_list) < api_item.index:
            init_list.append('        NULL')
        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    genapi.write_file(header_file, s)
    s = c_template % ',\n'.join(init_list)
    genapi.write_file(c_file, s)
    return targets

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    generate_api(outdir_abs)
if __name__ == '__main__':
    main()