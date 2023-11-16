import shapes_ext
import unittest
import numpy

class TestShapes(unittest.TestCase):

    def testShapes(self):
        if False:
            while True:
                i = 10
        a1 = numpy.array([(0, 1), (2, 3)])
        a1_shape = (1, 4)
        a1 = shapes_ext.reshape(a1, a1_shape)
        self.assertEqual(a1_shape, a1.shape)
if __name__ == '__main__':
    unittest.main()