import asyncio
import gc
import os
import sys
import threading
import warnings
import pytest
import numpy as np
from numpy.testing import extbuild, assert_warns, IS_WASM
from numpy._core.multiarray import get_handler_name

@pytest.fixture
def get_module(tmp_path):
    if False:
        while True:
            i = 10
    ' Add a memory policy that returns a false pointer 64 bytes into the\n    actual allocation, and fill the prefix with some text. Then check at each\n    memory manipulation that the prefix exists, to make sure all alloc/realloc/\n    free/calloc go via the functions here.\n    '
    if sys.platform.startswith('cygwin'):
        pytest.skip('link fails on cygwin')
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    functions = [('get_default_policy', 'METH_NOARGS', '\n             Py_INCREF(PyDataMem_DefaultHandler);\n             return PyDataMem_DefaultHandler;\n         '), ('set_secret_data_policy', 'METH_NOARGS', '\n             PyObject *secret_data =\n                 PyCapsule_New(&secret_data_handler, "mem_handler", NULL);\n             if (secret_data == NULL) {\n                 return NULL;\n             }\n             PyObject *old = PyDataMem_SetHandler(secret_data);\n             Py_DECREF(secret_data);\n             return old;\n         '), ('set_old_policy', 'METH_O', '\n             PyObject *old;\n             if (args != NULL && PyCapsule_CheckExact(args)) {\n                 old = PyDataMem_SetHandler(args);\n             }\n             else {\n                 old = PyDataMem_SetHandler(NULL);\n             }\n             return old;\n         '), ('get_array', 'METH_NOARGS', '\n            char *buf = (char *)malloc(20);\n            npy_intp dims[1];\n            dims[0] = 20;\n            PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);\n            return PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL,\n                                        buf, NPY_ARRAY_WRITEABLE, NULL);\n         '), ('set_own', 'METH_O', '\n            if (!PyArray_Check(args)) {\n                PyErr_SetString(PyExc_ValueError,\n                             "need an ndarray");\n                return NULL;\n            }\n            PyArray_ENABLEFLAGS((PyArrayObject*)args, NPY_ARRAY_OWNDATA);\n            // Maybe try this too?\n            // PyArray_BASE(PyArrayObject *)args) = NULL;\n            Py_RETURN_NONE;\n         '), ('get_array_with_base', 'METH_NOARGS', '\n            char *buf = (char *)malloc(20);\n            npy_intp dims[1];\n            dims[0] = 20;\n            PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);\n            PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims,\n                                                 NULL, buf,\n                                                 NPY_ARRAY_WRITEABLE, NULL);\n            if (arr == NULL) return NULL;\n            PyObject *obj = PyCapsule_New(buf, "buf capsule",\n                                          (PyCapsule_Destructor)&warn_on_free);\n            if (obj == NULL) {\n                Py_DECREF(arr);\n                return NULL;\n            }\n            if (PyArray_SetBaseObject((PyArrayObject *)arr, obj) < 0) {\n                Py_DECREF(arr);\n                Py_DECREF(obj);\n                return NULL;\n            }\n            return arr;\n\n         ')]
    prologue = '\n        #define NPY_TARGET_VERSION NPY_1_22_API_VERSION\n        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION\n        #include <numpy/arrayobject.h>\n        /*\n         * This struct allows the dynamic configuration of the allocator funcs\n         * of the `secret_data_allocator`. It is provided here for\n         * demonstration purposes, as a valid `ctx` use-case scenario.\n         */\n        typedef struct {\n            void *(*malloc)(size_t);\n            void *(*calloc)(size_t, size_t);\n            void *(*realloc)(void *, size_t);\n            void (*free)(void *);\n        } SecretDataAllocatorFuncs;\n\n        NPY_NO_EXPORT void *\n        shift_alloc(void *ctx, size_t sz) {\n            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;\n            char *real = (char *)funcs->malloc(sz + 64);\n            if (real == NULL) {\n                return NULL;\n            }\n            snprintf(real, 64, "originally allocated %ld", (unsigned long)sz);\n            return (void *)(real + 64);\n        }\n        NPY_NO_EXPORT void *\n        shift_zero(void *ctx, size_t sz, size_t cnt) {\n            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;\n            char *real = (char *)funcs->calloc(sz + 64, cnt);\n            if (real == NULL) {\n                return NULL;\n            }\n            snprintf(real, 64, "originally allocated %ld via zero",\n                     (unsigned long)sz);\n            return (void *)(real + 64);\n        }\n        NPY_NO_EXPORT void\n        shift_free(void *ctx, void * p, npy_uintp sz) {\n            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;\n            if (p == NULL) {\n                return ;\n            }\n            char *real = (char *)p - 64;\n            if (strncmp(real, "originally allocated", 20) != 0) {\n                fprintf(stdout, "uh-oh, unmatched shift_free, "\n                        "no appropriate prefix\\n");\n                /* Make C runtime crash by calling free on the wrong address */\n                funcs->free((char *)p + 10);\n                /* funcs->free(real); */\n            }\n            else {\n                npy_uintp i = (npy_uintp)atoi(real +20);\n                if (i != sz) {\n                    fprintf(stderr, "uh-oh, unmatched shift_free"\n                            "(ptr, %ld) but allocated %ld\\n", sz, i);\n                    /* This happens in some places, only print */\n                    funcs->free(real);\n                }\n                else {\n                    funcs->free(real);\n                }\n            }\n        }\n        NPY_NO_EXPORT void *\n        shift_realloc(void *ctx, void * p, npy_uintp sz) {\n            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;\n            if (p != NULL) {\n                char *real = (char *)p - 64;\n                if (strncmp(real, "originally allocated", 20) != 0) {\n                    fprintf(stdout, "uh-oh, unmatched shift_realloc\\n");\n                    return realloc(p, sz);\n                }\n                return (void *)((char *)funcs->realloc(real, sz + 64) + 64);\n            }\n            else {\n                char *real = (char *)funcs->realloc(p, sz + 64);\n                if (real == NULL) {\n                    return NULL;\n                }\n                snprintf(real, 64, "originally allocated "\n                         "%ld  via realloc", (unsigned long)sz);\n                return (void *)(real + 64);\n            }\n        }\n        /* As an example, we use the standard {m|c|re}alloc/free funcs. */\n        static SecretDataAllocatorFuncs secret_data_handler_ctx = {\n            malloc,\n            calloc,\n            realloc,\n            free\n        };\n        static PyDataMem_Handler secret_data_handler = {\n            "secret_data_allocator",\n            1,\n            {\n                &secret_data_handler_ctx, /* ctx */\n                shift_alloc,              /* malloc */\n                shift_zero,               /* calloc */\n                shift_realloc,            /* realloc */\n                shift_free                /* free */\n            }\n        };\n        void warn_on_free(void *capsule) {\n            PyErr_WarnEx(PyExc_UserWarning, "in warn_on_free", 1);\n            void * obj = PyCapsule_GetPointer(capsule,\n                                              PyCapsule_GetName(capsule));\n            free(obj);\n        };\n        '
    more_init = 'import_array();'
    try:
        import mem_policy
        return mem_policy
    except ImportError:
        pass
    return extbuild.build_and_import_extension('mem_policy', functions, prologue=prologue, include_dirs=[np.get_include()], build_dir=tmp_path, more_init=more_init)

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_set_policy(get_module):
    if False:
        return 10
    get_handler_name = np._core.multiarray.get_handler_name
    get_handler_version = np._core.multiarray.get_handler_version
    orig_policy_name = get_handler_name()
    a = np.arange(10).reshape((2, 5))
    assert get_handler_name(a) is None
    assert get_handler_version(a) is None
    assert get_handler_name(a.base) == orig_policy_name
    assert get_handler_version(a.base) == 1
    orig_policy = get_module.set_secret_data_policy()
    b = np.arange(10).reshape((2, 5))
    assert get_handler_name(b) is None
    assert get_handler_version(b) is None
    assert get_handler_name(b.base) == 'secret_data_allocator'
    assert get_handler_version(b.base) == 1
    if orig_policy_name == 'default_allocator':
        get_module.set_old_policy(None)
        assert get_handler_name() == 'default_allocator'
    else:
        get_module.set_old_policy(orig_policy)
        assert get_handler_name() == orig_policy_name

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_default_policy_singleton(get_module):
    if False:
        while True:
            i = 10
    get_handler_name = np._core.multiarray.get_handler_name
    orig_policy = get_module.set_old_policy(None)
    assert get_handler_name() == 'default_allocator'
    def_policy_1 = get_module.set_old_policy(None)
    assert get_handler_name() == 'default_allocator'
    def_policy_2 = get_module.set_old_policy(orig_policy)
    assert def_policy_1 is def_policy_2 is get_module.get_default_policy()

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_policy_propagation(get_module):
    if False:
        return 10

    class MyArr(np.ndarray):
        pass
    get_handler_name = np._core.multiarray.get_handler_name
    orig_policy_name = get_handler_name()
    a = np.arange(10).view(MyArr).reshape((2, 5))
    assert get_handler_name(a) is None
    assert a.flags.owndata is False
    assert get_handler_name(a.base) is None
    assert a.base.flags.owndata is False
    assert get_handler_name(a.base.base) == orig_policy_name
    assert a.base.base.flags.owndata is True

async def concurrent_context1(get_module, orig_policy_name, event):
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        assert get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        assert get_handler_name() == 'default_allocator'
    event.set()

async def concurrent_context2(get_module, orig_policy_name, event):
    await event.wait()
    assert get_handler_name() == orig_policy_name
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        assert get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        assert get_handler_name() == 'default_allocator'

async def async_test_context_locality(get_module):
    orig_policy_name = np._core.multiarray.get_handler_name()
    event = asyncio.Event()
    concurrent_task1 = asyncio.create_task(concurrent_context1(get_module, orig_policy_name, event))
    concurrent_task2 = asyncio.create_task(concurrent_context2(get_module, orig_policy_name, event))
    await concurrent_task1
    await concurrent_task2
    assert np._core.multiarray.get_handler_name() == orig_policy_name

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_context_locality(get_module):
    if False:
        i = 10
        return i + 15
    if sys.implementation.name == 'pypy' and sys.pypy_version_info[:3] < (7, 3, 6):
        pytest.skip('no context-locality support in PyPy < 7.3.6')
    asyncio.run(async_test_context_locality(get_module))

def concurrent_thread1(get_module, event):
    if False:
        for i in range(10):
            print('nop')
    get_module.set_secret_data_policy()
    assert np._core.multiarray.get_handler_name() == 'secret_data_allocator'
    event.set()

def concurrent_thread2(get_module, event):
    if False:
        return 10
    event.wait()
    assert np._core.multiarray.get_handler_name() == 'default_allocator'
    get_module.set_secret_data_policy()

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_thread_locality(get_module):
    if False:
        for i in range(10):
            print('nop')
    orig_policy_name = np._core.multiarray.get_handler_name()
    event = threading.Event()
    concurrent_task1 = threading.Thread(target=concurrent_thread1, args=(get_module, event))
    concurrent_task2 = threading.Thread(target=concurrent_thread2, args=(get_module, event))
    concurrent_task1.start()
    concurrent_task2.start()
    concurrent_task1.join()
    concurrent_task2.join()
    assert np._core.multiarray.get_handler_name() == orig_policy_name

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
@pytest.mark.skip(reason='too slow, see gh-23975')
def test_new_policy(get_module):
    if False:
        i = 10
        return i + 15
    a = np.arange(10)
    orig_policy_name = np._core.multiarray.get_handler_name(a)
    orig_policy = get_module.set_secret_data_policy()
    b = np.arange(10)
    assert np._core.multiarray.get_handler_name(b) == 'secret_data_allocator'
    if orig_policy_name == 'default_allocator':
        assert np._core.test('full', verbose=1, extra_argv=[])
        assert np.ma.test('full', verbose=1, extra_argv=[])
    get_module.set_old_policy(orig_policy)
    c = np.arange(10)
    assert np._core.multiarray.get_handler_name(c) == orig_policy_name

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
@pytest.mark.xfail(sys.implementation.name == 'pypy', reason='bad interaction between getenv and os.environ inside pytest')
@pytest.mark.parametrize('policy', ['0', '1', None])
def test_switch_owner(get_module, policy):
    if False:
        i = 10
        return i + 15
    a = get_module.get_array()
    assert np._core.multiarray.get_handler_name(a) is None
    get_module.set_own(a)
    if policy is None:
        policy = os.getenv('NUMPY_WARN_IF_NO_MEM_POLICY', '0') == '1'
        oldval = None
    else:
        policy = policy == '1'
        oldval = np._core._multiarray_umath._set_numpy_warn_if_no_mem_policy(policy)
    try:
        if policy:
            with assert_warns(RuntimeWarning) as w:
                del a
                gc.collect()
        else:
            del a
            gc.collect()
    finally:
        if oldval is not None:
            np._core._multiarray_umath._set_numpy_warn_if_no_mem_policy(oldval)

@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
def test_owner_is_base(get_module):
    if False:
        for i in range(10):
            print('nop')
    a = get_module.get_array_with_base()
    with pytest.warns(UserWarning, match='warn_on_free'):
        del a
        gc.collect()