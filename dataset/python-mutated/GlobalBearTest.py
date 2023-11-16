import unittest
from coalib.bears.GlobalBear import BEAR_KIND, GlobalBear
from coalib.settings.Section import Section

class GlobalBearTest(unittest.TestCase):

    def test_file_dict(self):
        if False:
            for i in range(10):
                print('nop')
        file_dict_0 = {'filename1': 'contents1', 'filename2': 'contents2'}
        file_dict_1 = file_dict_0.copy()
        bear = GlobalBear(file_dict_0, Section(''), None)
        self.assertEqual(bear.file_dict, file_dict_0)
        self.assertEqual(bear.file_dict, file_dict_1)

    def test_run_raises(self):
        if False:
            return 10
        bear = GlobalBear(None, Section(''), None)
        self.assertRaises(NotImplementedError, bear.run)

    def test_kind_is_staticmethod(self):
        if False:
            return 10
        self.assertEqual(GlobalBear.kind(), BEAR_KIND.GLOBAL)
        bear = GlobalBear(None, Section(''), None)
        self.assertEqual(bear.kind(), BEAR_KIND.GLOBAL)