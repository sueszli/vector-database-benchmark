import os
import json
import unittest
from unittest import mock
from pythonforandroid.bootstrap import Bootstrap
from pythonforandroid.distribution import Distribution
from pythonforandroid.recipe import Recipe
from pythonforandroid.util import BuildInterruptingException
from pythonforandroid.build import Context
dist_info_data = {'dist_name': 'sdl2_dist', 'bootstrap': 'sdl2', 'archs': ['armeabi', 'armeabi-v7a', 'x86', 'x86_64', 'arm64-v8a'], 'ndk_api': 21, 'use_setup_py': False, 'recipes': ['hostpython3', 'python3', 'sdl2', 'kivy', 'requests'], 'hostpython': '/some/fake/hostpython3', 'python_version': '3.7'}

class TestDistribution(unittest.TestCase):
    """
    An inherited class of `unittest.TestCase`to test the module
    :mod:`~pythonforandroid.distribution`.
    """
    TEST_ARCH = 'armeabi-v7a'

    def setUp(self):
        if False:
            while True:
                i = 10
        'Configure a :class:`~pythonforandroid.build.Context` so we can\n        perform our unittests'
        self.ctx = Context()
        self.ctx.ndk_api = 21
        self.ctx.android_api = 27
        self.ctx._sdk_dir = '/opt/android/android-sdk'
        self.ctx._ndk_dir = '/opt/android/android-ndk'
        self.ctx.setup_dirs(os.getcwd())
        self.ctx.recipe_build_order = ['hostpython3', 'python3', 'sdl2', 'kivy']

    def setUp_distribution_with_bootstrap(self, bs, **kwargs):
        if False:
            return 10
        'Extend the setUp by configuring a distribution, because some test\n        needs a distribution to be set to be properly tested'
        self.ctx.bootstrap = bs
        self.ctx.bootstrap.distribution = Distribution.get_distribution(self.ctx, name=kwargs.pop('name', 'test_prj'), recipes=kwargs.pop('recipes', ['python3', 'kivy']), archs=[self.TEST_ARCH], **kwargs)

    def tearDown(self):
        if False:
            return 10
        'Here we make sure that we reset a possible bootstrap created in\n        `setUp_distribution_with_bootstrap`'
        self.ctx.bootstrap = None

    def test_properties(self):
        if False:
            return 10
        'Test that some attributes has the expected result (for now, we check\n        that `__repr__` and `__str__` return the proper values'
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx))
        distribution = self.ctx.bootstrap.distribution
        self.assertEqual(self.ctx, distribution.ctx)
        expected_repr = '<Distribution: name test_prj with recipes (python3, kivy)>'
        self.assertEqual(distribution.__str__(), expected_repr)
        self.assertEqual(distribution.__repr__(), expected_repr)

    @mock.patch('pythonforandroid.distribution.exists')
    def test_folder_exist(self, mock_exists):
        if False:
            while True:
                i = 10
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.folder_exist` is\n        called once with the proper arguments.'
        mock_exists.return_value = False
        self.setUp_distribution_with_bootstrap(Bootstrap.get_bootstrap('sdl2', self.ctx))
        self.ctx.bootstrap.distribution.folder_exists()
        mock_exists.assert_called_with(self.ctx.bootstrap.distribution.dist_dir)

    @mock.patch('pythonforandroid.distribution.rmdir')
    def test_delete(self, mock_rmdir):
        if False:
            return 10
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.delete` is\n        called once with the proper arguments.'
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx))
        self.ctx.bootstrap.distribution.delete()
        mock_rmdir.assert_called_once_with(self.ctx.bootstrap.distribution.dist_dir)

    @mock.patch('pythonforandroid.distribution.exists')
    def test_get_distribution_no_name(self, mock_exists):
        if False:
            print('Hello World!')
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.get_distribution`\n        returns the proper result which should `unnamed_dist_1`.'
        mock_exists.return_value = False
        self.ctx.bootstrap = Bootstrap().get_bootstrap('sdl2', self.ctx)
        dist = Distribution.get_distribution(self.ctx, archs=[self.TEST_ARCH])
        self.assertEqual(dist.name, 'unnamed_dist_1')

    @mock.patch('pythonforandroid.util.chdir')
    @mock.patch('pythonforandroid.distribution.open', create=True)
    def test_save_info(self, mock_open_dist_info, mock_chdir):
        if False:
            while True:
                i = 10
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.save_info`\n        is called once with the proper arguments.'
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx))
        self.ctx.hostpython = '/some/fake/hostpython3'
        self.ctx.python_recipe = Recipe.get_recipe('python3', self.ctx)
        self.ctx.python_modules = ['requests']
        mock_open_dist_info.side_effect = [mock.mock_open(read_data=json.dumps(dist_info_data)).return_value]
        self.ctx.bootstrap.distribution.save_info('/fake_dir')
        mock_open_dist_info.assert_called_once_with('dist_info.json', 'w')
        mock_open_dist_info.reset_mock()

    @mock.patch('pythonforandroid.distribution.open', create=True)
    @mock.patch('pythonforandroid.distribution.exists')
    @mock.patch('pythonforandroid.distribution.glob.glob')
    def test_get_distributions(self, mock_glob, mock_exists, mock_open_dist_info):
        if False:
            print('Hello World!')
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.get_distributions`\n        returns some expected values:\n\n            - A list of instances of class\n              `~pythonforandroid.distribution.Distribution\n            - That one of the distributions returned in the result has the\n              proper values (`name`, `ndk_api` and `recipes`)\n        '
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx))
        mock_glob.return_value = ['sdl2-python3']
        mock_open_dist_info.side_effect = [mock.mock_open(read_data=json.dumps(dist_info_data)).return_value]
        dists = self.ctx.bootstrap.distribution.get_distributions(self.ctx)
        self.assertIsInstance(dists, list)
        self.assertEqual(len(dists), 1)
        self.assertIsInstance(dists[0], Distribution)
        self.assertEqual(dists[0].name, 'sdl2_dist')
        self.assertEqual(dists[0].dist_dir, 'sdl2-python3')
        self.assertEqual(dists[0].ndk_api, 21)
        self.assertEqual(dists[0].recipes, ['hostpython3', 'python3', 'sdl2', 'kivy', 'requests'])
        mock_open_dist_info.assert_called_with('sdl2-python3/dist_info.json')
        mock_open_dist_info.reset_mock()

    @mock.patch('pythonforandroid.distribution.open', create=True)
    @mock.patch('pythonforandroid.distribution.exists')
    @mock.patch('pythonforandroid.distribution.glob.glob')
    def test_get_distributions_error_ndk_api(self, mock_glob, mock_exists, mock_open_dist_info):
        if False:
            i = 10
            return i + 15
        'Test method\n        :meth:`~pythonforandroid.distribution.Distribution.get_distributions`\n        in case that `ndk_api` is not set..which should return a `None`.\n        '
        dist_info_data_no_ndk_api = dist_info_data.copy()
        dist_info_data_no_ndk_api.pop('ndk_api')
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx))
        mock_glob.return_value = ['sdl2-python3']
        mock_open_dist_info.side_effect = [mock.mock_open(read_data=json.dumps(dist_info_data_no_ndk_api)).return_value]
        dists = self.ctx.bootstrap.distribution.get_distributions(self.ctx)
        self.assertEqual(dists[0].ndk_api, None)
        mock_open_dist_info.assert_called_with('sdl2-python3/dist_info.json')
        mock_open_dist_info.reset_mock()

    @mock.patch('pythonforandroid.distribution.Distribution.get_distributions')
    @mock.patch('pythonforandroid.distribution.exists')
    @mock.patch('pythonforandroid.distribution.glob.glob')
    def test_get_distributions_error_ndk_api_mismatch(self, mock_glob, mock_exists, mock_get_dists):
        if False:
            return 10
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.get_distribution`\n        raises an error in case that we have some distribution already build,\n        with a given `name` and `ndk_api`, and we try to get another\n        distribution with the same `name` but different `ndk_api`.\n        '
        expected_dist = Distribution.get_distribution(self.ctx, name='test_prj', recipes=['python3', 'kivy'], archs=[self.TEST_ARCH])
        mock_get_dists.return_value = [expected_dist]
        mock_glob.return_value = ['sdl2-python3']
        with self.assertRaises(BuildInterruptingException) as e:
            self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx), allow_replace_dist=False, ndk_api=22)
        self.assertEqual(e.exception.args[0], 'Asked for dist with name test_prj with recipes (python3, kivy) and NDK API 22, but a dist with this name already exists and has either incompatible recipes (python3, kivy) or NDK API 21')

    def test_get_distributions_error_extra_dist_dirs(self):
        if False:
            return 10
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.get_distributions`\n        raises an exception of\n        :class:`~pythonforandroid.util.BuildInterruptingException` in case that\n        we supply the kwargs `extra_dist_dirs`.\n        '
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx))
        with self.assertRaises(BuildInterruptingException) as e:
            self.ctx.bootstrap.distribution.get_distributions(self.ctx, extra_dist_dirs=['/fake/extra/dist_dirs'])
        self.assertEqual(e.exception.args[0], 'extra_dist_dirs argument to get_distributions is not yet implemented')

    @mock.patch('pythonforandroid.distribution.Distribution.get_distributions')
    def test_get_distributions_possible_dists(self, mock_get_dists):
        if False:
            return 10
        'Test that method\n        :meth:`~pythonforandroid.distribution.Distribution.get_distributions`\n        returns the proper\n        `:class:`~pythonforandroid.distribution.Distribution` in case that we\n        already have it build and we request the same\n        `:class:`~pythonforandroid.distribution.Distribution`.\n        '
        expected_dist = Distribution.get_distribution(self.ctx, name='test_prj', recipes=['python3', 'kivy'], archs=[self.TEST_ARCH])
        mock_get_dists.return_value = [expected_dist]
        self.setUp_distribution_with_bootstrap(Bootstrap().get_bootstrap('sdl2', self.ctx), name='test_prj')
        dists = self.ctx.bootstrap.distribution.get_distributions(self.ctx)
        self.assertEqual(dists[0], expected_dist)