import unittest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.widget import Widget
from kivy.graphics import Fbo, Color, Rectangle

class FboTest(Widget):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(FboTest, self).__init__(**kwargs)
        self.positions = [(260.0, 260.0), (192.0, 192.0), (96.0, 192.0), (192.0, 96.0), (96.0, 96.0), (32.0, 192.0), (192.0, 32.0), (32.0, 32.0)]
        self.fbo = Fbo(size=(256, 256))
        with self.fbo:
            Color(0.56789, 0, 0, 1)
            Rectangle(size=(256, 64))
            Color(0, 0.56789, 0, 1)
            Rectangle(size=(64, 256))
            Color(0.56789, 0, 0, 0.5)
            Rectangle(pos=(64, 64), size=(192, 64))
            Color(0, 0.56789, 0, 0.5)
            Rectangle(pos=(64, 64), size=(64, 192))
        self.fbo.draw()

class FBOPy2Py3TestCase(GraphicUnitTest):

    def test_fbo_get_pixel_color(self):
        if False:
            print('Hello World!')
        fbow = FboTest()
        render_error = 2
        values = [(tuple, int, (0, 0, 0, 0)), (list, int, [0, 0, 0, 0]), (list, int, [0, 72, 0, 128]), (list, int, [72, 0, 0, 128]), (list, int, [36, 72, 0, 255]), (list, int, [0, 145, 0, 255]), (list, int, [145, 0, 0, 255]), (list, int, [0, 145, 0, 255])]
        for (i, pos) in enumerate(fbow.positions):
            c = fbow.fbo.get_pixel_color(pos[0], pos[1])
            self.assertTrue(isinstance(c, values[i][0]))
            for v in c:
                self.assertTrue(isinstance(v, values[i][1]))
            for (j, val) in enumerate(c):
                self.assertAlmostEqual(val, values[i][2][j], delta=render_error)
if __name__ == '__main__':
    unittest.main()