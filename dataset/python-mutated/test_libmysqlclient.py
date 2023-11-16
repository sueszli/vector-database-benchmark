import unittest
from unittest import mock
from tests.recipes.recipe_lib_test import BaseTestForCmakeRecipe

class TestLibmysqlclientRecipe(BaseTestForCmakeRecipe, unittest.TestCase):
    """
    An unittest for recipe :mod:`~pythonforandroid.recipes.libmysqlclient`
    """
    recipe_name = 'libmysqlclient'

    @mock.patch('pythonforandroid.recipes.libmysqlclient.sh.rm')
    @mock.patch('pythonforandroid.recipes.libmysqlclient.sh.cp')
    @mock.patch('pythonforandroid.util.chdir')
    @mock.patch('pythonforandroid.build.ensure_dir')
    @mock.patch('shutil.which')
    def test_build_arch(self, mock_shutil_which, mock_ensure_dir, mock_current_directory, mock_sh_cp, mock_sh_rm):
        if False:
            for i in range(10):
                print('nop')
        super().test_build_arch()
        mock_sh_cp.assert_called()
        mock_sh_rm.assert_called()