import unittest
from .. import model_generator

class ModelGeneratorTest(unittest.TestCase):

    def test_qualifier(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(model_generator.qualifier('/root', '/root'), '.')
        self.assertEqual(model_generator.qualifier('/root', '/root/a.py'), 'a')
        self.assertEqual(model_generator.qualifier('/root', '/root/dir/a.py'), 'dir.a')
        self.assertEqual(model_generator.qualifier('/root', '/root/dir/__init__.py'), 'dir')
        self.assertEqual(model_generator.qualifier('/root', '/root/a.pyi'), 'a')
        self.assertEqual(model_generator.qualifier('/root', '/root/dir/a.pyi'), 'dir.a')
        self.assertEqual(model_generator.qualifier('/root', '/root/dir/__init__.pyi'), 'dir')