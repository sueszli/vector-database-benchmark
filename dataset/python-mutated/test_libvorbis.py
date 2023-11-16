import unittest
from unittest import mock
from tests.recipes.recipe_lib_test import BaseTestForMakeRecipe

class TestLibvorbisRecipe(BaseTestForMakeRecipe, unittest.TestCase):
    """
    An unittest for recipe :mod:`~pythonforandroid.recipes.libvorbis`
    """
    recipe_name = 'libvorbis'
    sh_command_calls = ['./configure']
    extra_env_flags = {'CFLAGS': 'libogg/include'}

    @mock.patch('pythonforandroid.recipes.libvorbis.sh.cp')
    @mock.patch('pythonforandroid.util.chdir')
    @mock.patch('pythonforandroid.build.ensure_dir')
    @mock.patch('shutil.which')
    def test_build_arch(self, mock_shutil_which, mock_ensure_dir, mock_current_directory, mock_sh_cp):
        if False:
            i = 10
            return i + 15
        super().test_build_arch()
        mock_sh_cp.assert_called()