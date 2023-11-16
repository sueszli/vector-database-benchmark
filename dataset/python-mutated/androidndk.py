import sys
import os

class AndroidNDK:
    """
    This class is used to get the current NDK information.
    """
    ndk_dir = ''

    def __init__(self, ndk_dir):
        if False:
            for i in range(10):
                print('nop')
        self.ndk_dir = ndk_dir

    @property
    def host_tag(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the host tag for the current system.\n        Note: The host tag is ``darwin-x86_64`` even on Apple Silicon macs.\n        '
        return f'{sys.platform}-x86_64'

    @property
    def llvm_prebuilt_dir(self):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.ndk_dir, 'toolchains', 'llvm', 'prebuilt', self.host_tag)

    @property
    def llvm_bin_dir(self):
        if False:
            print('Hello World!')
        return os.path.join(self.llvm_prebuilt_dir, 'bin')

    @property
    def clang(self):
        if False:
            return 10
        return os.path.join(self.llvm_bin_dir, 'clang')

    @property
    def clang_cxx(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(self.llvm_bin_dir, 'clang++')

    @property
    def llvm_binutils_prefix(self):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.llvm_bin_dir, 'llvm-')

    @property
    def llvm_ar(self):
        if False:
            i = 10
            return i + 15
        return f'{self.llvm_binutils_prefix}ar'

    @property
    def llvm_ranlib(self):
        if False:
            while True:
                i = 10
        return f'{self.llvm_binutils_prefix}ranlib'

    @property
    def llvm_objcopy(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.llvm_binutils_prefix}objcopy'

    @property
    def llvm_objdump(self):
        if False:
            print('Hello World!')
        return f'{self.llvm_binutils_prefix}objdump'

    @property
    def llvm_readelf(self):
        if False:
            return 10
        return f'{self.llvm_binutils_prefix}readelf'

    @property
    def llvm_strip(self):
        if False:
            i = 10
            return i + 15
        return f'{self.llvm_binutils_prefix}strip'

    @property
    def sysroot(self):
        if False:
            print('Hello World!')
        return os.path.join(self.llvm_prebuilt_dir, 'sysroot')

    @property
    def sysroot_include_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(self.sysroot, 'usr', 'include')

    @property
    def sysroot_lib_dir(self):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.sysroot, 'usr', 'lib')

    @property
    def libcxx_include_dir(self):
        if False:
            while True:
                i = 10
        return os.path.join(self.sysroot_include_dir, 'c++', 'v1')