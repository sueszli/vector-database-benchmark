import unittest
import pyxel

class TestPyxel(unittest.TestCase):

    def setUpClass():
        if False:
            for i in range(10):
                print('nop')
        pyxel.init(300, 300, 'hoge')

    def test_title(self):
        if False:
            return 10
        pyxel.title('hoge')

    def test_colors(self):
        if False:
            i = 10
            return i + 15
        default_colors = [0, 2831199, 8265842, 1676700, 9128018, 3759256, 11125247, 15658734, 13899884, 13861953, 15319899, 7390889, 7771870, 10724259, 16750488, 15583152] * 2
        self.assertEqual(pyxel.colors.to_list(), default_colors)
        modified_colors = default_colors[:]
        modified_colors[0:4] = [1118481, 2236962, 3355443, 4473924]
        pyxel.colors.from_list([1118481, 2236962, 3355443, 4473924])
        self.assertEqual(pyxel.colors.to_list(), modified_colors)
        extended_colors = default_colors[:] + [16777215]
        pyxel.colors.from_list(extended_colors)
        extended_colors.pop()
        self.assertEqual(pyxel.colors.to_list(), extended_colors)
        self.assertEqual(pyxel.colors[0], 0)
        pyxel.colors[0] = 1122867
        self.assertEqual(pyxel.colors[0], 1122867)

    def test_cls(self):
        if False:
            i = 10
            return i + 15
        pyxel.cls(3)

    def test_input_text(self):
        if False:
            i = 10
            return i + 15
        pyxel.input_text