import gc
import platform
import unittest
from prometheus_client import CollectorRegistry, GCCollector
SKIP = platform.python_implementation() != 'CPython'

@unittest.skipIf(SKIP, 'Test requires CPython')
class TestGCCollector(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        gc.disable()
        gc.collect()
        self.registry = CollectorRegistry()

    def test_working(self):
        if False:
            for i in range(10):
                print('nop')
        GCCollector(registry=self.registry)
        self.registry.collect()
        before = self.registry.get_sample_value('python_gc_objects_collected_total', labels={'generation': '0'})
        a = []
        a.append(a)
        del a
        b = []
        b.append(b)
        del b
        gc.collect(0)
        self.registry.collect()
        after = self.registry.get_sample_value('python_gc_objects_collected_total', labels={'generation': '0'})
        self.assertEqual(2, after - before)
        self.assertEqual(0, self.registry.get_sample_value('python_gc_objects_uncollectable_total', labels={'generation': '0'}))

    def test_empty(self):
        if False:
            print('Hello World!')
        GCCollector(registry=self.registry)
        self.registry.collect()
        before = self.registry.get_sample_value('python_gc_objects_collected_total', labels={'generation': '0'})
        gc.collect(0)
        self.registry.collect()
        after = self.registry.get_sample_value('python_gc_objects_collected_total', labels={'generation': '0'})
        self.assertEqual(0, after - before)

    def tearDown(self):
        if False:
            print('Hello World!')
        gc.enable()