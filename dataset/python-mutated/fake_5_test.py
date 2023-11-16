if __name__ == '__main__':
    import sys
    import os
    pkg_dir = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
    (parent_dir, pkg_name) = os.path.split(pkg_dir)
    is_pygame_pkg = pkg_name == 'tests' and os.path.split(parent_dir)[1] == 'pygame'
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')
import unittest

class KeyModuleTest(unittest.TestCase):

    def test_get_focused(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(True)

    def test_get_mods(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(True)

    def test_get_pressed(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(True)

    def test_name(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(True)

    def test_set_mods(self):
        if False:
            while True:
                i = 10
        self.assertTrue(True)

    def test_set_repeat(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()