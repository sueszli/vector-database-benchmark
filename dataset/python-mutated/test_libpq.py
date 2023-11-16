import unittest
from unittest import mock
from tests.recipes.recipe_lib_test import BaseTestForMakeRecipe

class TestLibpqRecipe(BaseTestForMakeRecipe, unittest.TestCase):
    """
    An unittest for recipe :mod:`~pythonforandroid.recipes.libpq`
    """
    recipe_name = 'libpq'
    sh_command_calls = ['./configure']

    @mock.patch('pythonforandroid.recipes.libpq.sh.cp')
    @mock.patch('pythonforandroid.util.chdir')
    @mock.patch('pythonforandroid.build.ensure_dir')
    @mock.patch('shutil.which')
    def test_build_arch(self, mock_shutil_which, mock_ensure_dir, mock_current_directory, mock_sh_cp):
        if False:
            while True:
                i = 10
        super().test_build_arch()
        mock_sh_cp.assert_called()