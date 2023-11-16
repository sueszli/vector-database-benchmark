import unittest

class TestLoadAttrCache(unittest.TestCase):

    def test_descriptor_added_after_optimization(self):
        if False:
            for i in range(10):
                print('nop')

        class Descriptor:
            pass

        class C:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 1
            x = Descriptor()

        def f(o):
            if False:
                i = 10
                return i + 15
            return o.x
        o = C()
        for i in range(1025):
            assert f(o) == 1
        Descriptor.__get__ = lambda self, instance, value: 2
        Descriptor.__set__ = lambda *args: None
        self.assertEqual(f(o), 2)