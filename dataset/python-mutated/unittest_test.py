"""Test the unittest itself."""
import unittest
g_count = 0
g_setUpClass_count = 0
g_tearDownClass_count = 0
g_setUp_count = 0
g_tearDown_count = 0

class Test(unittest.TestCase):
    count = 0

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        global g_setUpClass_count
        g_setUpClass_count += 1

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        global g_tearDownClass_count
        g_tearDownClass_count += 1

    def setUp(self):
        if False:
            i = 10
            return i + 15
        global g_setUp_count
        g_setUp_count += 1

    def tearDown(self):
        if False:
            while True:
                i = 10
        global g_tearDown_count
        g_tearDown_count += 1

    def test_unittest1(self):
        if False:
            while True:
                i = 10
        global g_count
        g_count += 1
        self.count += 1
        self.assertEqual(g_count, 1)
        self.assertEqual(self.count, 1)
        self.assertEqual(g_setUpClass_count, 1)
        self.assertEqual(g_tearDownClass_count, 0)

    def test_unittest2(self):
        if False:
            i = 10
            return i + 15
        global g_count
        g_count += 1
        self.count += 1
        self.assertEqual(g_count, 2)
        self.assertEqual(self.count, 1)
        self.assertEqual(g_setUpClass_count, 1)
        self.assertEqual(g_tearDownClass_count, 0)

    def test_unittest3(self):
        if False:
            print('Hello World!')
        self.assertEqual(g_setUp_count, 3)
        self.assertEqual(g_tearDown_count, 2)
        self.assertEqual(g_setUpClass_count, 1)
        self.assertEqual(g_tearDownClass_count, 0)
if __name__ == '__main__':
    unittest.main()