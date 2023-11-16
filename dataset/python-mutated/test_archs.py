import os
import unittest
from os import environ
from unittest import mock
from pythonforandroid.bootstrap import Bootstrap
from pythonforandroid.distribution import Distribution
from pythonforandroid.recipe import Recipe
from pythonforandroid.build import Context
from pythonforandroid.util import BuildInterruptingException
from pythonforandroid.archs import Arch, ArchARM, ArchARMv7_a, ArchAarch_64, Archx86, Archx86_64
from pythonforandroid.androidndk import AndroidNDK
expected_env_gcc_keys = {'CFLAGS', 'LDFLAGS', 'CXXFLAGS', 'CC', 'CXX', 'LDSHARED', 'STRIP', 'MAKE', 'READELF', 'BUILDLIB_PATH', 'PATH', 'ARCH', 'NDK_API'}

class ArchSetUpBaseClass(object):
    """
    An class object which is intended to be used as a base class to configure
    an inherited class of `unittest.TestCase`. This class will override the
    `setUp` method.
    """
    ctx = None
    expected_compiler = ''
    TEST_ARCH = 'armeabi-v7a'

    def setUp(self):
        if False:
            return 10
        self.ctx = Context()
        self.ctx.ndk_api = 21
        self.ctx.android_api = 27
        self.ctx._sdk_dir = '/opt/android/android-sdk'
        self.ctx._ndk_dir = '/opt/android/android-ndk'
        self.ctx.ndk = AndroidNDK(self.ctx._ndk_dir)
        self.ctx.setup_dirs(os.getcwd())
        self.ctx.bootstrap = Bootstrap().get_bootstrap('sdl2', self.ctx)
        self.ctx.bootstrap.distribution = Distribution.get_distribution(self.ctx, name='sdl2', recipes=['python3', 'kivy'], archs=[self.TEST_ARCH])
        self.ctx.python_recipe = Recipe.get_recipe('python3', self.ctx)
        self.expected_compiler = f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ctx.ndk.host_tag}/bin/clang'

class TestArch(ArchSetUpBaseClass, unittest.TestCase):
    """
    An inherited class of `ArchSetUpBaseClass` and `unittest.TestCase` which
    will be used to perform tests for the base class
    :class:`~pythonforandroid.archs.Arch`.
    """

    def test_arch(self):
        if False:
            return 10
        arch = Arch(self.ctx)
        self.assertEqual(arch.__str__(), arch.arch)
        self.assertEqual(arch.target, 'None21')
        self.assertIsNone(arch.command_prefix)
        self.assertIsInstance(arch.include_dirs, list)

class TestArchARM(ArchSetUpBaseClass, unittest.TestCase):
    """
    An inherited class of `ArchSetUpBaseClass` and `unittest.TestCase` which
    will be used to perform tests for :class:`~pythonforandroid.archs.ArchARM`.
    """

    @mock.patch('shutil.which')
    @mock.patch('pythonforandroid.build.ensure_dir')
    def test_arch_arm(self, mock_ensure_dir, mock_shutil_which):
        if False:
            return 10
        "\n        Test that class :class:`~pythonforandroid.archs.ArchARM` returns some\n        expected attributes and environment variables.\n\n        .. note::\n            Here we mock two methods:\n\n                - `ensure_dir` because we don't want to create any directory\n                - `shutil.which` because otherwise we will\n                  get an error when trying to find the compiler (we are setting\n                  some fake paths for our android sdk and ndk so probably will\n                  not exist)\n\n        "
        mock_shutil_which.return_value = self.expected_compiler
        mock_ensure_dir.return_value = True
        arch = ArchARM(self.ctx)
        self.assertEqual(arch.arch, 'armeabi')
        self.assertEqual(arch.__str__(), 'armeabi')
        self.assertEqual(arch.command_prefix, 'arm-linux-androideabi')
        self.assertEqual(arch.target, 'armv7a-linux-androideabi21')
        arch = ArchARM(self.ctx)
        env = arch.get_env()
        self.assertIsInstance(env, dict)
        self.assertEqual(expected_env_gcc_keys, set(env.keys()) & expected_env_gcc_keys)
        mock_shutil_which.assert_called_once_with(self.expected_compiler, path=environ['PATH'])
        self.assertEqual(env['CC'].split()[0], self.expected_compiler)
        self.assertEqual(env['CXX'].split()[0], self.expected_compiler + '++')
        self.assertEqual(env['STRIP'].split()[0], os.path.join(self.ctx._ndk_dir, f'toolchains/llvm/prebuilt/{self.ctx.ndk.host_tag}/bin', 'llvm-strip'))
        self.assertEqual(env['READELF'].split()[0], os.path.join(self.ctx._ndk_dir, f'toolchains/llvm/prebuilt/{self.ctx.ndk.host_tag}/bin', 'llvm-readelf'))
        self.assertIn(env['CFLAGS'], env['CC'])
        self.ctx.ccache = '/usr/bin/ccache'
        env = arch.get_env(with_flags_in_cc=False)
        self.assertNotIn(env['CFLAGS'], env['CC'])
        self.assertEqual(env['USE_CCACHE'], '1')
        self.assertEqual(env['NDK_CCACHE'], '/usr/bin/ccache')
        mock_shutil_which.return_value = None
        with self.assertRaises(BuildInterruptingException) as e:
            arch.get_env()
        self.assertEqual(e.exception.args[0], "Couldn't find executable for CC. This indicates a problem locating the {expected_compiler} executable in the Android NDK, not that you don't have a normal compiler installed. Exiting.".format(expected_compiler=self.expected_compiler))

class TestArchARMv7a(ArchSetUpBaseClass, unittest.TestCase):
    """
    An inherited class of `ArchSetUpBaseClass` and `unittest.TestCase` which
    will be used to perform tests for
    :class:`~pythonforandroid.archs.ArchARMv7_a`.
    """

    @mock.patch('shutil.which')
    @mock.patch('pythonforandroid.build.ensure_dir')
    def test_arch_armv7a(self, mock_ensure_dir, mock_shutil_which):
        if False:
            while True:
                i = 10
        '\n        Test that class :class:`~pythonforandroid.archs.ArchARMv7_a` returns\n        some expected attributes and environment variables.\n\n        .. note::\n            Here we mock the same functions than\n            :meth:`TestArchARM.test_arch_arm`.\n            This has to be done because here we tests the `get_env` with clang\n\n        '
        mock_shutil_which.return_value = self.expected_compiler
        mock_ensure_dir.return_value = True
        arch = ArchARMv7_a(self.ctx)
        self.assertEqual(arch.arch, 'armeabi-v7a')
        self.assertEqual(arch.__str__(), 'armeabi-v7a')
        self.assertEqual(arch.command_prefix, 'arm-linux-androideabi')
        self.assertEqual(arch.target, 'armv7a-linux-androideabi21')
        env = arch.get_env()
        mock_shutil_which.assert_called_once_with(self.expected_compiler, path=environ['PATH'])
        self.assertEqual(env['CC'].split()[0], '{ndk_dir}/toolchains/llvm/prebuilt/{host_tag}/bin/clang'.format(ndk_dir=self.ctx._ndk_dir, host_tag=self.ctx.ndk.host_tag))
        self.assertEqual(env['CXX'].split()[0], '{ndk_dir}/toolchains/llvm/prebuilt/{host_tag}/bin/clang++'.format(ndk_dir=self.ctx._ndk_dir, host_tag=self.ctx.ndk.host_tag))
        self.assertIn(' -march=armv7-a -mfloat-abi=softfp -mfpu=vfp -mthumb', env['CFLAGS'])

class TestArchX86(ArchSetUpBaseClass, unittest.TestCase):
    """
    An inherited class of `ArchSetUpBaseClass` and `unittest.TestCase` which
    will be used to perform tests for :class:`~pythonforandroid.archs.Archx86`.
    """

    @mock.patch('shutil.which')
    @mock.patch('pythonforandroid.build.ensure_dir')
    def test_arch_x86(self, mock_ensure_dir, mock_shutil_which):
        if False:
            i = 10
            return i + 15
        "\n        Test that class :class:`~pythonforandroid.archs.Archx86` returns\n        some expected attributes and environment variables.\n\n        .. note::\n            Here we mock the same functions than\n            :meth:`TestArchARM.test_arch_arm` plus `glob`, so we make sure that\n            the glob result is the expected even if the folder doesn't exist,\n            which is probably the case. This has to be done because here we\n            tests the `get_env` with clang\n        "
        mock_shutil_which.return_value = self.expected_compiler
        mock_ensure_dir.return_value = True
        arch = Archx86(self.ctx)
        self.assertEqual(arch.arch, 'x86')
        self.assertEqual(arch.__str__(), 'x86')
        self.assertEqual(arch.command_prefix, 'i686-linux-android')
        self.assertEqual(arch.target, 'i686-linux-android21')
        env = arch.get_env()
        mock_shutil_which.assert_called_once_with(self.expected_compiler, path=environ['PATH'])
        self.assertIn(' -march=i686 -mssse3 -mfpmath=sse -m32', env['CFLAGS'])

class TestArchX86_64(ArchSetUpBaseClass, unittest.TestCase):
    """
    An inherited class of `ArchSetUpBaseClass` and `unittest.TestCase` which
    will be used to perform tests for
    :class:`~pythonforandroid.archs.Archx86_64`.
    """

    @mock.patch('shutil.which')
    @mock.patch('pythonforandroid.build.ensure_dir')
    def test_arch_x86_64(self, mock_ensure_dir, mock_shutil_which):
        if False:
            i = 10
            return i + 15
        "\n        Test that class :class:`~pythonforandroid.archs.Archx86_64` returns\n        some expected attributes and environment variables.\n\n        .. note::\n            Here we mock the same functions than\n            :meth:`TestArchARM.test_arch_arm` plus `glob`, so we make sure that\n            the glob result is the expected even if the folder doesn't exist,\n            which is probably the case. This has to be done because here we\n            tests the `get_env` with clang\n        "
        mock_shutil_which.return_value = self.expected_compiler
        mock_ensure_dir.return_value = True
        arch = Archx86_64(self.ctx)
        self.assertEqual(arch.arch, 'x86_64')
        self.assertEqual(arch.__str__(), 'x86_64')
        self.assertEqual(arch.command_prefix, 'x86_64-linux-android')
        self.assertEqual(arch.target, 'x86_64-linux-android21')
        env = arch.get_env()
        mock_shutil_which.assert_called_once_with(self.expected_compiler, path=environ['PATH'])
        mock_shutil_which.assert_called_once()
        self.assertIn(' -march=x86-64 -msse4.2 -mpopcnt -m64', env['CFLAGS'])

class TestArchAArch64(ArchSetUpBaseClass, unittest.TestCase):
    """
    An inherited class of `ArchSetUpBaseClass` and `unittest.TestCase` which
    will be used to perform tests for
    :class:`~pythonforandroid.archs.ArchAarch_64`.
    """

    @mock.patch('shutil.which')
    @mock.patch('pythonforandroid.build.ensure_dir')
    def test_arch_aarch_64(self, mock_ensure_dir, mock_shutil_which):
        if False:
            return 10
        "\n        Test that class :class:`~pythonforandroid.archs.ArchAarch_64` returns\n        some expected attributes and environment variables.\n\n        .. note::\n            Here we mock the same functions than\n            :meth:`TestArchARM.test_arch_arm` plus `glob`, so we make sure that\n            the glob result is the expected even if the folder doesn't exist,\n            which is probably the case. This has to be done because here we\n            tests the `get_env` with clang\n        "
        mock_shutil_which.return_value = self.expected_compiler
        mock_ensure_dir.return_value = True
        arch = ArchAarch_64(self.ctx)
        self.assertEqual(arch.arch, 'arm64-v8a')
        self.assertEqual(arch.__str__(), 'arm64-v8a')
        self.assertEqual(arch.command_prefix, 'aarch64-linux-android')
        self.assertEqual(arch.target, 'aarch64-linux-android21')
        env = arch.get_env()
        mock_shutil_which.assert_called_once_with(self.expected_compiler, path=environ['PATH'])
        for flag in {'CFLAGS', 'CXXFLAGS', 'CC', 'CXX'}:
            self.assertIn('-march=armv8-a', env[flag])