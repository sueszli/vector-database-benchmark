import sys
import pytest
import numpy as np
from numpy.testing import extbuild, IS_WASM

@pytest.fixture
def get_module(tmp_path):
    if False:
        while True:
            i = 10
    ' Some codes to generate data and manage temporary buffers use when\n    sharing with numpy via the array interface protocol.\n    '
    if not sys.platform.startswith('linux'):
        pytest.skip('link fails on cygwin')
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    prologue = '\n        #include <Python.h>\n        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION\n        #include <numpy/arrayobject.h>\n        #include <stdio.h>\n        #include <math.h>\n\n        NPY_NO_EXPORT\n        void delete_array_struct(PyObject *cap) {\n\n            /* get the array interface structure */\n            PyArrayInterface *inter = (PyArrayInterface*)\n                PyCapsule_GetPointer(cap, NULL);\n\n            /* get the buffer by which data was shared */\n            double *ptr = (double*)PyCapsule_GetContext(cap);\n\n            /* for the purposes of the regression test set the elements\n               to nan */\n            for (npy_intp i = 0; i < inter->shape[0]; ++i)\n                ptr[i] = nan("");\n\n            /* free the shared buffer */\n            free(ptr);\n\n            /* free the array interface structure */\n            free(inter->shape);\n            free(inter);\n\n            fprintf(stderr, "delete_array_struct\\ncap = %ld inter = %ld"\n                " ptr = %ld\\n", (long)cap, (long)inter, (long)ptr);\n        }\n        '
    functions = [('new_array_struct', 'METH_VARARGS', '\n\n            long long n_elem = 0;\n            double value = 0.0;\n\n            if (!PyArg_ParseTuple(args, "Ld", &n_elem, &value)) {\n                Py_RETURN_NONE;\n            }\n\n            /* allocate and initialize the data to share with numpy */\n            long long n_bytes = n_elem*sizeof(double);\n            double *data = (double*)malloc(n_bytes);\n\n            if (!data) {\n                PyErr_Format(PyExc_MemoryError,\n                    "Failed to malloc %lld bytes", n_bytes);\n\n                Py_RETURN_NONE;\n            }\n\n            for (long long i = 0; i < n_elem; ++i) {\n                data[i] = value;\n            }\n\n            /* calculate the shape and stride */\n            int nd = 1;\n\n            npy_intp *ss = (npy_intp*)malloc(2*nd*sizeof(npy_intp));\n            npy_intp *shape = ss;\n            npy_intp *stride = ss + nd;\n\n            shape[0] = n_elem;\n            stride[0] = sizeof(double);\n\n            /* construct the array interface */\n            PyArrayInterface *inter = (PyArrayInterface*)\n                malloc(sizeof(PyArrayInterface));\n\n            memset(inter, 0, sizeof(PyArrayInterface));\n\n            inter->two = 2;\n            inter->nd = nd;\n            inter->typekind = \'f\';\n            inter->itemsize = sizeof(double);\n            inter->shape = shape;\n            inter->strides = stride;\n            inter->data = data;\n            inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_NOTSWAPPED |\n                           NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;\n\n            /* package into a capsule */\n            PyObject *cap = PyCapsule_New(inter, NULL, delete_array_struct);\n\n            /* save the pointer to the data */\n            PyCapsule_SetContext(cap, data);\n\n            fprintf(stderr, "new_array_struct\\ncap = %ld inter = %ld"\n                " ptr = %ld\\n", (long)cap, (long)inter, (long)data);\n\n            return cap;\n        ')]
    more_init = 'import_array();'
    try:
        import array_interface_testing
        return array_interface_testing
    except ImportError:
        pass
    return extbuild.build_and_import_extension('array_interface_testing', functions, prologue=prologue, include_dirs=[np.get_include()], build_dir=tmp_path, more_init=more_init)

@pytest.mark.slow
def test_cstruct(get_module):
    if False:
        print('Hello World!')

    class data_source:
        """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

        def __init__(self, size, value):
            if False:
                return 10
            self.size = size
            self.value = value

        @property
        def __array_struct__(self):
            if False:
                return 10
            return get_module.new_array_struct(self.size, self.value)
    stderr = sys.__stderr__
    expected_value = -3.1415
    multiplier = -10000.0
    stderr.write(' ---- create an object to share data ---- \n')
    buf = data_source(256, expected_value)
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- share data via the array interface protocol ---- \n')
    arr = np.array(buf, copy=False)
    stderr.write('arr.__array_interface___ = %s\n' % str(arr.__array_interface__))
    stderr.write('arr.base = %s\n' % str(arr.base))
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- destroy the object that shared data ---- \n')
    buf = None
    stderr.write(' ---- OK!\n\n')
    assert np.allclose(arr, expected_value)
    stderr.write(' ---- read shared data ---- \n')
    stderr.write('arr = %s\n' % str(arr))
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- modify shared data ---- \n')
    arr *= multiplier
    expected_value *= multiplier
    stderr.write('arr.__array_interface___ = %s\n' % str(arr.__array_interface__))
    stderr.write('arr.base = %s\n' % str(arr.base))
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- read modified shared data ---- \n')
    stderr.write('arr = %s\n' % str(arr))
    stderr.write(' ---- OK!\n\n')
    assert np.allclose(arr, expected_value)
    stderr.write(' ---- free shared data ---- \n')
    arr = None
    stderr.write(' ---- OK!\n\n')