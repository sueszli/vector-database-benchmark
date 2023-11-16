import os
import argparse
import genapi
from genapi import TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi
import numpy_api
h_template = '\n#if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)\n\ntypedef struct {\n        PyObject_HEAD\n        npy_bool obval;\n} PyBoolScalarObject;\n\nextern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;\nextern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];\n\n%s\n\n#else\n\n#if defined(PY_ARRAY_UNIQUE_SYMBOL)\n    #define PyArray_API PY_ARRAY_UNIQUE_SYMBOL\n    #define _NPY_VERSION_CONCAT_HELPER2(x, y) x ## y\n    #define _NPY_VERSION_CONCAT_HELPER(arg) \\\n        _NPY_VERSION_CONCAT_HELPER2(arg, PyArray_RUNTIME_VERSION)\n    #define PyArray_RUNTIME_VERSION \\\n        _NPY_VERSION_CONCAT_HELPER(PY_ARRAY_UNIQUE_SYMBOL)\n#endif\n\n#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)\nextern void **PyArray_API;\nextern int PyArray_RUNTIME_VERSION;\n#else\n#if defined(PY_ARRAY_UNIQUE_SYMBOL)\nvoid **PyArray_API;\nint PyArray_RUNTIME_VERSION;\n#else\nstatic void **PyArray_API = NULL;\nstatic int PyArray_RUNTIME_VERSION = 0;\n#endif\n#endif\n\n%s\n\n#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)\nstatic int\n_import_array(void)\n{\n  int st;\n  PyObject *numpy = PyImport_ImportModule("numpy._core._multiarray_umath");\n  if (numpy == NULL && PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {\n    PyErr_Clear();\n    numpy = PyImport_ImportModule("numpy.core._multiarray_umath");\n  }\n\n  if (numpy == NULL) {\n      return -1;\n  }\n\n  PyObject *c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");\n  Py_DECREF(numpy);\n  if (c_api == NULL) {\n      return -1;\n  }\n\n  if (!PyCapsule_CheckExact(c_api)) {\n      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");\n      Py_DECREF(c_api);\n      return -1;\n  }\n  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);\n  Py_DECREF(c_api);\n  if (PyArray_API == NULL) {\n      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");\n      return -1;\n  }\n\n  /*\n   * Perform runtime check of C API version.  As of now NumPy 2.0 is ABI\n   * backwards compatible (in the exposed feature subset!) for all practical\n   * purposes.\n   */\n  if (NPY_VERSION < PyArray_GetNDArrayCVersion()) {\n      PyErr_Format(PyExc_RuntimeError, "module compiled against "\\\n             "ABI version 0x%%x but this version of numpy is 0x%%x", \\\n             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());\n      return -1;\n  }\n  PyArray_RUNTIME_VERSION = (int)PyArray_GetNDArrayCFeatureVersion();\n  if (NPY_FEATURE_VERSION > PyArray_RUNTIME_VERSION) {\n      PyErr_Format(PyExc_RuntimeError, "module compiled against "\\\n             "API version 0x%%x but this version of numpy is 0x%%x . "\\\n             "Check the section C-API incompatibility at the "\\\n             "Troubleshooting ImportError section at "\\\n             "https://numpy.org/devdocs/user/troubleshooting-importerror.html"\\\n             "#c-api-incompatibility "\\\n              "for indications on how to solve this problem .", \\\n             (int)NPY_FEATURE_VERSION, PyArray_RUNTIME_VERSION);\n      return -1;\n  }\n\n  /*\n   * Perform runtime check of endianness and check it matches the one set by\n   * the headers (npy_endian.h) as a safeguard\n   */\n  st = PyArray_GetEndianness();\n  if (st == NPY_CPU_UNKNOWN_ENDIAN) {\n      PyErr_SetString(PyExc_RuntimeError,\n                      "FATAL: module compiled as unknown endian");\n      return -1;\n  }\n#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN\n  if (st != NPY_CPU_BIG) {\n      PyErr_SetString(PyExc_RuntimeError,\n                      "FATAL: module compiled as big endian, but "\n                      "detected different endianness at runtime");\n      return -1;\n  }\n#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN\n  if (st != NPY_CPU_LITTLE) {\n      PyErr_SetString(PyExc_RuntimeError,\n                      "FATAL: module compiled as little endian, but "\n                      "detected different endianness at runtime");\n      return -1;\n  }\n#endif\n\n  return 0;\n}\n\n#define import_array() { \\\n  if (_import_array() < 0) { \\\n    PyErr_Print(); \\\n    PyErr_SetString( \\\n        PyExc_ImportError, \\\n        "numpy._core.multiarray failed to import" \\\n    ); \\\n    return NULL; \\\n  } \\\n}\n\n#define import_array1(ret) { \\\n  if (_import_array() < 0) { \\\n    PyErr_Print(); \\\n    PyErr_SetString( \\\n        PyExc_ImportError, \\\n        "numpy._core.multiarray failed to import" \\\n    ); \\\n    return ret; \\\n  } \\\n}\n\n#define import_array2(msg, ret) { \\\n  if (_import_array() < 0) { \\\n    PyErr_Print(); \\\n    PyErr_SetString(PyExc_ImportError, msg); \\\n    return ret; \\\n  } \\\n}\n\n#endif\n\n#endif\n'
c_template = '\n/* These pointers will be stored in the C-object for use in other\n    extension modules\n*/\n\nvoid *PyArray_API[] = {\n%s\n};\n'

def generate_api(output_dir, force=False):
    if False:
        while True:
            i = 10
    basename = 'multiarray_api'
    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    targets = (h_file, c_file)
    sources = numpy_api.multiarray_api
    do_generate_api(targets, sources)
    return targets

def do_generate_api(targets, sources):
    if False:
        while True:
            i = 10
    header_file = targets[0]
    c_file = targets[1]
    global_vars = sources[0]
    scalar_bool_values = sources[1]
    types_api = sources[2]
    multiarray_funcs = sources[3]
    multiarray_api = sources[:]
    module_list = []
    extension_list = []
    init_list = []
    multiarray_api_index = genapi.merge_api_dicts(multiarray_api)
    genapi.check_api_dict(multiarray_api_index)
    numpyapi_list = genapi.get_api_functions('NUMPY_API', multiarray_funcs)
    api_name = 'PyArray_API'
    multiarray_api_dict = {}
    for f in numpyapi_list:
        name = f.name
        index = multiarray_funcs[name][0]
        annotations = multiarray_funcs[name][1:]
        multiarray_api_dict[f.name] = FunctionApi(f.name, index, annotations, f.return_type, f.args, api_name)
    for (name, val) in global_vars.items():
        (index, type) = val
        multiarray_api_dict[name] = GlobalVarApi(name, index, type, api_name)
    for (name, val) in scalar_bool_values.items():
        index = val[0]
        multiarray_api_dict[name] = BoolValuesApi(name, index, api_name)
    for (name, val) in types_api.items():
        index = val[0]
        internal_type = None if len(val) == 1 else val[1]
        multiarray_api_dict[name] = TypeApi(name, index, 'PyTypeObject', api_name, internal_type)
    if len(multiarray_api_dict) != len(multiarray_api_index):
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        raise AssertionError('Multiarray API size mismatch - index has extra keys {}, dict has extra keys {}'.format(keys_index - keys_dict, keys_dict - keys_index))
    extension_list = []
    for (name, index) in genapi.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
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
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output directory')
    parser.add_argument('-i', '--ignore', type=str, help='An ignored input - may be useful to add a dependency between custom targets')
    args = parser.parse_args()
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    generate_api(outdir_abs)
if __name__ == '__main__':
    main()