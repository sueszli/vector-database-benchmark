import unittest
from unittest import mock
from pythonforandroid.androidndk import AndroidNDK

class TestAndroidNDK(unittest.TestCase):
    """
    An inherited class of `unittest.TestCase`to test the module
    :mod:`~pythonforandroid.androidndk`.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        'Configure a :class:`~pythonforandroid.androidndk.AndroidNDK` so we can\n        perform our unittests'
        self.ndk = AndroidNDK('/opt/android/android-ndk')

    @mock.patch('sys.platform', 'linux')
    def test_host_tag_linux(self):
        if False:
            print('Hello World!')
        'Test the `host_tag` property of the :class:`~pythonforandroid.androidndk.AndroidNDK`\n        class when the host is Linux.'
        self.assertEqual(self.ndk.host_tag, 'linux-x86_64')

    @mock.patch('sys.platform', 'darwin')
    def test_host_tag_darwin(self):
        if False:
            i = 10
            return i + 15
        'Test the `host_tag` property of the :class:`~pythonforandroid.androidndk.AndroidNDK`\n        class when the host is Darwin.'
        self.assertEqual(self.ndk.host_tag, 'darwin-x86_64')

    def test_llvm_prebuilt_dir(self):
        if False:
            i = 10
            return i + 15
        'Test the `llvm_prebuilt_dir` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_prebuilt_dir, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}')

    def test_llvm_bin_dir(self):
        if False:
            while True:
                i = 10
        'Test the `llvm_bin_dir` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_bin_dir, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin')

    def test_clang(self):
        if False:
            i = 10
            return i + 15
        'Test the `clang` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.clang, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/clang')

    def test_clang_cxx(self):
        if False:
            i = 10
            return i + 15
        'Test the `clang_cxx` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.clang_cxx, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/clang++')

    def test_llvm_ar(self):
        if False:
            i = 10
            return i + 15
        'Test the `llvm_ar` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_ar, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/llvm-ar')

    def test_llvm_ranlib(self):
        if False:
            while True:
                i = 10
        'Test the `llvm_ranlib` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_ranlib, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/llvm-ranlib')

    def test_llvm_objcopy(self):
        if False:
            return 10
        'Test the `llvm_objcopy` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_objcopy, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/llvm-objcopy')

    def test_llvm_objdump(self):
        if False:
            while True:
                i = 10
        'Test the `llvm_objdump` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_objdump, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/llvm-objdump')

    def test_llvm_readelf(self):
        if False:
            i = 10
            return i + 15
        'Test the `llvm_readelf` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_readelf, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/llvm-readelf')

    def test_llvm_strip(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the `llvm_strip` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.llvm_strip, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/bin/llvm-strip')

    def test_sysroot(self):
        if False:
            while True:
                i = 10
        'Test the `sysroot` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.sysroot, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/sysroot')

    def test_sysroot_include_dir(self):
        if False:
            while True:
                i = 10
        'Test the `sysroot_include_dir` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.sysroot_include_dir, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/sysroot/usr/include')

    def test_sysroot_lib_dir(self):
        if False:
            i = 10
            return i + 15
        'Test the `sysroot_lib_dir` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.sysroot_lib_dir, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/sysroot/usr/lib')

    def test_libcxx_include_dir(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the `libcxx_include_dir` property of the\n        :class:`~pythonforandroid.androidndk.AndroidNDK` class.'
        self.assertEqual(self.ndk.libcxx_include_dir, f'/opt/android/android-ndk/toolchains/llvm/prebuilt/{self.ndk.host_tag}/sysroot/usr/include/c++/v1')