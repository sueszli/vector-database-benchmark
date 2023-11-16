from runner.koan import *
from .triangle import *

class AboutTriangleProject2(Koan):

    def test_illegal_triangles_throw_exceptions(self):
        if False:
            return 10
        with self.assertRaises(TriangleError):
            triangle(0, 0, 0)
        with self.assertRaises(TriangleError):
            triangle(3, 4, -5)
        with self.assertRaises(TriangleError):
            triangle(1, 1, 3)
        with self.assertRaises(TriangleError):
            triangle(2, 5, 2)