import shutil
import os
import tempfile
import time
import Cython.Build.Dependencies
import Cython.Utils
from Cython.TestUtils import CythonTest

def fresh_cythonize(*args, **kwargs):
    if False:
        print('Hello World!')
    Cython.Utils.clear_function_caches()
    Cython.Build.Dependencies._dep_tree = None
    Cython.Build.Dependencies.cythonize(*args, **kwargs)

class TestRecythonize(CythonTest):

    def setUp(self):
        if False:
            print('Hello World!')
        CythonTest.setUp(self)
        self.temp_dir = tempfile.mkdtemp(prefix='recythonize-test', dir='TEST_TMP' if os.path.isdir('TEST_TMP') else None)

    def tearDown(self):
        if False:
            print('Hello World!')
        CythonTest.tearDown(self)
        shutil.rmtree(self.temp_dir)

    def test_recythonize_pyx_on_pxd_change(self):
        if False:
            while True:
                i = 10
        src_dir = tempfile.mkdtemp(prefix='src', dir=self.temp_dir)
        a_pxd = os.path.join(src_dir, 'a.pxd')
        a_pyx = os.path.join(src_dir, 'a.pyx')
        a_c = os.path.join(src_dir, 'a.c')
        dep_tree = Cython.Build.Dependencies.create_dependency_tree()
        with open(a_pxd, 'w') as f:
            f.write('cdef int value\n')
        with open(a_pyx, 'w') as f:
            f.write('value = 1\n')
        self.assertEqual({a_pxd, a_pyx}, dep_tree.all_dependencies(a_pyx))
        fresh_cythonize(a_pyx)
        time.sleep(1)
        with open(a_c) as f:
            a_c_contents1 = f.read()
        with open(a_pxd, 'w') as f:
            f.write('cdef double value\n')
        fresh_cythonize(a_pyx)
        with open(a_c) as f:
            a_c_contents2 = f.read()
        self.assertTrue('__pyx_v_1a_value = 1;' in a_c_contents1)
        self.assertFalse('__pyx_v_1a_value = 1;' in a_c_contents2)
        self.assertTrue('__pyx_v_1a_value = 1.0;' in a_c_contents2)
        self.assertFalse('__pyx_v_1a_value = 1.0;' in a_c_contents1)

    def test_recythonize_py_on_pxd_change(self):
        if False:
            print('Hello World!')
        src_dir = tempfile.mkdtemp(prefix='src', dir=self.temp_dir)
        a_pxd = os.path.join(src_dir, 'a.pxd')
        a_py = os.path.join(src_dir, 'a.py')
        a_c = os.path.join(src_dir, 'a.c')
        dep_tree = Cython.Build.Dependencies.create_dependency_tree()
        with open(a_pxd, 'w') as f:
            f.write('cdef int value\n')
        with open(a_py, 'w') as f:
            f.write('value = 1\n')
        self.assertEqual({a_pxd, a_py}, dep_tree.all_dependencies(a_py))
        fresh_cythonize(a_py)
        time.sleep(1)
        with open(a_c) as f:
            a_c_contents1 = f.read()
        with open(a_pxd, 'w') as f:
            f.write('cdef double value\n')
        fresh_cythonize(a_py)
        with open(a_c) as f:
            a_c_contents2 = f.read()
        self.assertTrue('__pyx_v_1a_value = 1;' in a_c_contents1)
        self.assertFalse('__pyx_v_1a_value = 1;' in a_c_contents2)
        self.assertTrue('__pyx_v_1a_value = 1.0;' in a_c_contents2)
        self.assertFalse('__pyx_v_1a_value = 1.0;' in a_c_contents1)

    def test_recythonize_pyx_on_dep_pxd_change(self):
        if False:
            i = 10
            return i + 15
        src_dir = tempfile.mkdtemp(prefix='src', dir=self.temp_dir)
        a_pxd = os.path.join(src_dir, 'a.pxd')
        a_pyx = os.path.join(src_dir, 'a.pyx')
        b_pyx = os.path.join(src_dir, 'b.pyx')
        b_c = os.path.join(src_dir, 'b.c')
        dep_tree = Cython.Build.Dependencies.create_dependency_tree()
        with open(a_pxd, 'w') as f:
            f.write('cdef int value\n')
        with open(a_pyx, 'w') as f:
            f.write('value = 1\n')
        with open(b_pyx, 'w') as f:
            f.write('cimport a\n' + 'a.value = 2\n')
        self.assertEqual({a_pxd, b_pyx}, dep_tree.all_dependencies(b_pyx))
        fresh_cythonize([a_pyx, b_pyx])
        time.sleep(1)
        with open(b_c) as f:
            b_c_contents1 = f.read()
        with open(a_pxd, 'w') as f:
            f.write('cdef double value\n')
        fresh_cythonize([a_pyx, b_pyx])
        with open(b_c) as f:
            b_c_contents2 = f.read()
        self.assertTrue('__pyx_v_1a_value = 2;' in b_c_contents1)
        self.assertFalse('__pyx_v_1a_value = 2;' in b_c_contents2)
        self.assertTrue('__pyx_v_1a_value = 2.0;' in b_c_contents2)
        self.assertFalse('__pyx_v_1a_value = 2.0;' in b_c_contents1)

    def test_recythonize_py_on_dep_pxd_change(self):
        if False:
            return 10
        src_dir = tempfile.mkdtemp(prefix='src', dir=self.temp_dir)
        a_pxd = os.path.join(src_dir, 'a.pxd')
        a_pyx = os.path.join(src_dir, 'a.pyx')
        b_pxd = os.path.join(src_dir, 'b.pxd')
        b_py = os.path.join(src_dir, 'b.py')
        b_c = os.path.join(src_dir, 'b.c')
        dep_tree = Cython.Build.Dependencies.create_dependency_tree()
        with open(a_pxd, 'w') as f:
            f.write('cdef int value\n')
        with open(a_pyx, 'w') as f:
            f.write('value = 1\n')
        with open(b_pxd, 'w') as f:
            f.write('cimport a\n')
        with open(b_py, 'w') as f:
            f.write('a.value = 2\n')
        self.assertEqual({a_pxd, b_pxd, b_py}, dep_tree.all_dependencies(b_py))
        fresh_cythonize([a_pyx, b_py])
        time.sleep(1)
        with open(b_c) as f:
            b_c_contents1 = f.read()
        with open(a_pxd, 'w') as f:
            f.write('cdef double value\n')
        fresh_cythonize([a_pyx, b_py])
        with open(b_c) as f:
            b_c_contents2 = f.read()
        self.assertTrue('__pyx_v_1a_value = 2;' in b_c_contents1)
        self.assertFalse('__pyx_v_1a_value = 2;' in b_c_contents2)
        self.assertTrue('__pyx_v_1a_value = 2.0;' in b_c_contents2)
        self.assertFalse('__pyx_v_1a_value = 2.0;' in b_c_contents1)