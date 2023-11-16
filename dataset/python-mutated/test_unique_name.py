import unittest
from paddle import base

class TestUniqueName(unittest.TestCase):

    def test_guard(self):
        if False:
            i = 10
            return i + 15
        with base.unique_name.guard():
            name_1 = base.unique_name.generate('')
        with base.unique_name.guard():
            name_2 = base.unique_name.generate('')
        self.assertEqual(name_1, name_2)
        with base.unique_name.guard('A'):
            name_1 = base.unique_name.generate('')
        with base.unique_name.guard('B'):
            name_2 = base.unique_name.generate('')
        self.assertNotEqual(name_1, name_2)

    def test_generate(self):
        if False:
            while True:
                i = 10
        with base.unique_name.guard():
            name1 = base.unique_name.generate('fc')
            name2 = base.unique_name.generate('fc')
            name3 = base.unique_name.generate('tmp')
            self.assertNotEqual(name1, name2)
            self.assertEqual(name1[-2:], name3[-2:])

class TestImperativeUniqueName(unittest.TestCase):

    def test_name_generator(self):
        if False:
            return 10
        with base.dygraph.guard():
            tracer = base.framework._dygraph_tracer()
            tmp_var_0 = tracer._generate_unique_name()
            self.assertEqual(tmp_var_0, 'dygraph_tmp_0')
            tmp_var_1 = tracer._generate_unique_name('dygraph_tmp')
            self.assertEqual(tmp_var_1, 'dygraph_tmp_1')
if __name__ == '__main__':
    unittest.main()