import gc
import test.test_gc
import unittest
try:
    import cinder
except ImportError:
    raise unittest.SkipTest('Tests CinderX features')

def _restore_parallel_gc(settings):
    if False:
        while True:
            i = 10
    cinder.disable_parallel_gc()
    if settings is not None:
        cinder.enable_parallel_gc(settings['min_generation'], settings['num_threads'])

class ParallelGCAPITests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.old_par_gc_settings = cinder.get_parallel_gc_settings()
        cinder.disable_parallel_gc()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        _restore_parallel_gc(self.old_par_gc_settings)

    def test_get_settings_when_disabled(self):
        if False:
            while True:
                i = 10
        self.assertEqual(cinder.get_parallel_gc_settings(), None)

    def test_get_settings_when_enabled(self):
        if False:
            while True:
                i = 10
        cinder.enable_parallel_gc(2, 8)
        settings = cinder.get_parallel_gc_settings()
        expected = {'min_generation': 2, 'num_threads': 8}
        self.assertEqual(settings, expected)

    def test_set_invalid_generation(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'invalid generation'):
            cinder.enable_parallel_gc(4, 8)

    def test_set_invalid_num_threads(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'invalid num_threads'):
            cinder.enable_parallel_gc(2, -1)

class ParallelGCTests(test.test_gc.GCTests):
    pass

class ParallelGCCallbackTests(test.test_gc.GCCallbackTests):

    @unittest.skip('Tests implementation details of serial collector')
    def test_refcount_errors(self):
        if False:
            i = 10
            return i + 15
        pass

class ParallelGCFinalizationTests(test.test_gc.PythonFinalizationTests):
    pass

def setUpModule():
    if False:
        print('Hello World!')
    test.test_gc.setUpModule()
    global old_par_gc_settings
    old_par_gc_settings = cinder.get_parallel_gc_settings()
    cinder.enable_parallel_gc(0, 8)

def tearDownModule():
    if False:
        i = 10
        return i + 15
    test.test_gc.tearDownModule()
    global old_par_gc_settings
    _restore_parallel_gc(old_par_gc_settings)
if __name__ == '__main__':
    unittest.main()