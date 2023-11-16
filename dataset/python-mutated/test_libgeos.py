import unittest
from unittest import mock
from tests.recipes.recipe_lib_test import BaseTestForCmakeRecipe

class TestLibgeosRecipe(BaseTestForCmakeRecipe, unittest.TestCase):
    """
    An unittest for recipe :mod:`~pythonforandroid.recipes.libgeos`
    """
    recipe_name = 'libgeos'

    @mock.patch('pythonforandroid.util.makedirs')
    @mock.patch('pythonforandroid.util.chdir')
    @mock.patch('pythonforandroid.build.ensure_dir')
    @mock.patch('shutil.which')
    def test_build_arch(self, mock_shutil_which, mock_ensure_dir, mock_current_directory, mock_makedirs):
        if False:
            i = 10
            return i + 15
        super().test_build_arch()
        mock_makedirs.assert_called()