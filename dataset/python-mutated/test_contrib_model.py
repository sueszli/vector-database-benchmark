import unittest
from qlib.contrib.model import all_model_classes

class TestAllFlow(unittest.TestCase):

    def test_0_initialize(self):
        if False:
            return 10
        num = 0
        for model_class in all_model_classes:
            if model_class is not None:
                model = model_class()
                num += 1
        print('There are {:}/{:} valid models in total.'.format(num, len(all_model_classes)))

def suite():
    if False:
        return 10
    _suite = unittest.TestSuite()
    _suite.addTest(TestAllFlow('test_0_initialize'))
    return _suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())