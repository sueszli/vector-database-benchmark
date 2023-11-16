import unittest
import pickle
import sys
import tempfile
from pathlib import Path

class TestUnpickleDeletedModule(unittest.TestCase):

    def test_loading_pickle_with_no_module(self):
        if False:
            return 10
        'Create a module that uses Numba, import a function from it.\n        Then delete the module and pickle the function. The function\n        should load from the pickle without a problem.\n\n        Note - This is a simplified version of how Numba might be used\n        on a distributed system using e.g. dask distributed. With the\n        pickle being sent to the worker but not the original module.\n        '
        source = '\n'.join(['from numba import vectorize', "@vectorize(['float64(float64)'])", 'def inc1(x):', '    return x + 1'])
        modname = 'tmp_module'
        with tempfile.TemporaryDirectory() as tmp_dir:
            sys.path.append(tmp_dir)
            filename = Path(f'{tmp_dir}/{modname}.py')
            f = open(filename, 'a')
            f.write(source)
            f.close()
            from tmp_module import inc1
        del sys.modules[modname]
        pkl = pickle.dumps(inc1)
        f = pickle.loads(pkl)
        self.assertEqual(f(2), 3)