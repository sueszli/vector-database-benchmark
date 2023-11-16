import unittest
from os.path import join
from tests.recipes.recipe_lib_test import BaseTestForMakeRecipe

class TestLibLzmaRecipe(BaseTestForMakeRecipe, unittest.TestCase):
    """TestCase for recipe :mod:`~pythonforandroid.recipes.liblzma`."""
    recipe_name = 'liblzma'
    sh_command_calls = ['./autogen.sh', 'autoreconf', './configure']

    def test_get_library_includes(self):
        if False:
            i = 10
            return i + 15
        '\n        Test :meth:`~pythonforandroid.recipes.liblzma.get_library_includes`.\n        '
        recipe_build_dir = self.recipe.get_build_dir(self.arch.arch)
        self.assertEqual(self.recipe.get_library_includes(self.arch), f" -I{join(recipe_build_dir, 'p4a_install/include')}")

    def test_get_library_ldflags(self):
        if False:
            while True:
                i = 10
        '\n        Test :meth:`~pythonforandroid.recipes.liblzma.get_library_ldflags`.\n        '
        recipe_build_dir = self.recipe.get_build_dir(self.arch.arch)
        self.assertEqual(self.recipe.get_library_ldflags(self.arch), f" -L{join(recipe_build_dir, 'p4a_install/lib')}")

    def test_link_libs_flags(self):
        if False:
            while True:
                i = 10
        '\n        Test :meth:`~pythonforandroid.recipes.liblzma.get_library_libs_flag`.\n        '
        self.assertEqual(self.recipe.get_library_libs_flag(), ' -llzma')

    def test_install_dir_not_named_install(self):
        if False:
            while True:
                i = 10
        '\n        Tests that the install directory is not named ``install``.\n\n        liblzma already have a file named ``INSTALL`` in its source directory.\n        On case-insensitive filesystems, using a folder named ``install`` will\n        cause a conflict. (See issue: #2343).\n\n        WARNING: This test is quite flaky, but should be enough to\n        ensure that someone in the future will not accidentally rename\n        the install directory without seeing this test to fail.\n        '
        liblzma_install_dir = self.recipe.built_libraries['liblzma.so']
        self.assertNotIn('install', liblzma_install_dir.split('/'))