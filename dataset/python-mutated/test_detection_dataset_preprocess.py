import os
import unittest

class Test_Preprocess(unittest.TestCase):

    def test_local_convert(self):
        if False:
            print('Hello World!')
        os.system('python full_pascalvoc_test_preprocess.py --choice=local')

    def test_online_convert(self):
        if False:
            return 10
        os.system('python full_pascalvoc_test_preprocess.py --choice=VOC_test_2007')
if __name__ == '__main__':
    unittest.main()