"""Configure build environment for certain Intel platforms."""
import argparse
import os
import subprocess
BASIC_BUILD_OPTS = ['--cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0', '--copt=-O3']
SECURE_BUILD_OPTS = ['--copt=-Wformat', '--copt=-Wformat-security', '--copt=-fstack-protector', '--copt=-fPIC', '--copt=-fpic', '--linkopt=-znoexecstack', '--linkopt=-zrelro', '--linkopt=-znow', '--linkopt=-fstack-protector']

class IntelPlatform(object):
    min_gcc_major_version_ = 0
    min_gcc_minor_version_ = 0
    host_gcc_major_version_ = 0
    host_gcc_minor_version_ = 0
    BAZEL_PREFIX_ = '--copt='
    ARCH_PREFIX_ = '-march='
    FLAG_PREFIX_ = '-m'

    def __init__(self, min_gcc_major_version, min_gcc_minor_version):
        if False:
            return 10
        self.min_gcc_minor_version_ = min_gcc_minor_version
        self.min_gcc_major_version_ = min_gcc_major_version

    def set_host_gcc_version(self, gcc_major_version, gcc_minor_version):
        if False:
            i = 10
            return i + 15
        if gcc_major_version < self.min_gcc_major_version_:
            print('Your MAJOR version of GCC is too old: {}; it must be at least {}.{}'.format(gcc_major_version, self.min_gcc_major_version_, self.min_gcc_minor_version_))
            return False
        elif gcc_major_version == self.min_gcc_major_version_ and gcc_minor_version < self.min_gcc_minor_version_:
            print('Your MINOR version of GCC is too old: {}; it must be at least {}.{}'.format(gcc_minor_version, self.min_gcc_major_version_, self.min_gcc_minor_version_))
            return False
        print('gcc version OK: {}.{}'.format(gcc_major_version, gcc_minor_version))
        self.host_gcc_major_version_ = gcc_major_version
        self.host_gcc_minor_version_ = gcc_minor_version
        return True

    def get_bazel_gcc_flags(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self)

    def use_old_arch_names(self, gcc_new_march_major_version, gcc_new_march_minor_version):
        if False:
            for i in range(10):
                print('nop')
        if self.host_gcc_major_version_ < gcc_new_march_major_version:
            return True
        elif self.host_gcc_major_version_ == gcc_new_march_major_version and self.host_gcc_minor_version_ < gcc_new_march_minor_version:
            return True
        return False

class NehalemPlatform(IntelPlatform):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        IntelPlatform.__init__(self, 4, 8)

    def get_bazel_gcc_flags(self):
        if False:
            return 10
        NEHALEM_ARCH_OLD = 'corei7'
        NEHALEM_ARCH_NEW = 'nehalem'
        if self.use_old_arch_names(4, 9):
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + NEHALEM_ARCH_OLD + ' '
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + NEHALEM_ARCH_NEW + ' '

class SandyBridgePlatform(IntelPlatform):

    def __init__(self):
        if False:
            while True:
                i = 10
        IntelPlatform.__init__(self, 4, 8)

    def get_bazel_gcc_flags(self):
        if False:
            i = 10
            return i + 15
        SANDYBRIDGE_ARCH_OLD = 'corei7-avx'
        SANDYBRIDGE_ARCH_NEW = 'sandybridge'
        if self.use_old_arch_names(4, 9):
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + SANDYBRIDGE_ARCH_OLD + ' '
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + SANDYBRIDGE_ARCH_NEW + ' '

class HaswellPlatform(IntelPlatform):

    def __init__(self):
        if False:
            print('Hello World!')
        IntelPlatform.__init__(self, 4, 8)

    def get_bazel_gcc_flags(self):
        if False:
            print('Hello World!')
        HASWELL_ARCH_OLD = 'core-avx2'
        HASWELL_ARCH_NEW = 'haswell'
        POPCNT_FLAG = 'popcnt'
        if self.use_old_arch_names(4, 9):
            ret_val = self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + HASWELL_ARCH_OLD + ' '
            return ret_val + self.BAZEL_PREFIX_ + self.FLAG_PREFIX_ + POPCNT_FLAG + ' '
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + HASWELL_ARCH_NEW + ' '

class SkylakePlatform(IntelPlatform):

    def __init__(self):
        if False:
            while True:
                i = 10
        IntelPlatform.__init__(self, 4, 9)

    def get_bazel_gcc_flags(self):
        if False:
            for i in range(10):
                print('nop')
        SKYLAKE_ARCH_OLD = 'broadwell'
        SKYLAKE_ARCH_NEW = 'skylake-avx512'
        AVX512_FLAGS = ['avx512f', 'avx512cd']
        if self.use_old_arch_names(6, 1):
            ret_val = self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + SKYLAKE_ARCH_OLD + ' '
            for flag in AVX512_FLAGS:
                ret_val += self.BAZEL_PREFIX_ + self.FLAG_PREFIX_ + flag + ' '
            return ret_val
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + SKYLAKE_ARCH_NEW + ' '

class CascadelakePlatform(IntelPlatform):

    def __init__(self):
        if False:
            print('Hello World!')
        IntelPlatform.__init__(self, 8, 3)

    def get_bazel_gcc_flags(self):
        if False:
            while True:
                i = 10
        CASCADELAKE_ARCH_OLD = 'skylake-avx512'
        CASCADELAKE_ARCH_NEW = 'cascadelake'
        VNNI_FLAG = 'avx512vnni'
        if IntelPlatform.use_old_arch_names(self, 9, 1):
            ret_val = self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + CASCADELAKE_ARCH_OLD + ' '
            return ret_val + self.BAZEL_PREFIX_ + self.FLAG_PREFIX_ + VNNI_FLAG + ' '
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + CASCADELAKE_ARCH_NEW + ' '

class IcelakeClientPlatform(IntelPlatform):

    def __init__(self):
        if False:
            return 10
        IntelPlatform.__init__(self, 8, 4)

    def get_bazel_gcc_flags(self):
        if False:
            print('Hello World!')
        ICELAKE_ARCH_OLD = 'skylake-avx512'
        ICELAKE_ARCH_NEW = 'icelake-client'
        AVX512_FLAGS = ['avx512f', 'avx512cd']
        if IntelPlatform.use_old_arch_names(self, 8, 4):
            ret_val = self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + ICELAKE_ARCH_OLD + ' '
            for flag in AVX512_FLAGS:
                ret_val += self.BAZEL_PREFIX_ + self.FLAG_PREFIX_ + flag + ' '
            return ret_val
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + ICELAKE_ARCH_NEW + ' '

class IcelakeServerPlatform(IntelPlatform):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        IntelPlatform.__init__(self, 8, 4)

    def get_bazel_gcc_flags(self):
        if False:
            print('Hello World!')
        ICELAKE_ARCH_OLD = 'skylake-avx512'
        ICELAKE_ARCH_NEW = 'icelake-server'
        AVX512_FLAGS = ['avx512f', 'avx512cd']
        if IntelPlatform.use_old_arch_names(self, 8, 4):
            ret_val = self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + ICELAKE_ARCH_OLD + ' '
            for flag in AVX512_FLAGS:
                ret_val += self.BAZEL_PREFIX_ + self.FLAG_PREFIX_ + flag + ' '
            return ret_val
        else:
            return self.BAZEL_PREFIX_ + self.ARCH_PREFIX_ + ICELAKE_ARCH_NEW + ' '

class BuildEnvSetter(object):
    """Prepares the proper environment settings for various Intel platforms."""
    default_platform_ = 'haswell'
    PLATFORMS_ = {'nehalem': NehalemPlatform(), 'sandybridge': SandyBridgePlatform(), 'haswell': HaswellPlatform(), 'skylake': SkylakePlatform(), 'cascadelake': CascadelakePlatform(), 'icelake-client': IcelakeClientPlatform(), 'icelake-server': IcelakeServerPlatform()}

    def __init__(self):
        if False:
            while True:
                i = 10
        self.args = None
        self.bazel_flags_ = 'build '
        self.target_platform_ = None

    def get_gcc_version(self):
        if False:
            for i in range(10):
                print('nop')
        gcc_major_version = 0
        gcc_minor_version = 0
        gcc_path = ''
        gcc_path_cmd = 'command -v gcc'
        try:
            gcc_path = subprocess.check_output(gcc_path_cmd, shell=True, stderr=subprocess.STDOUT).strip()
            print('gcc located here: {}'.format(gcc_path))
            if not os.access(gcc_path, os.F_OK | os.X_OK):
                raise ValueError('{} does not exist or is not executable.'.format(gcc_path))
            gcc_output = subprocess.check_output([gcc_path, '-dumpfullversion', '-dumpversion'], stderr=subprocess.STDOUT).strip()
            if isinstance(gcc_output, bytes):
                gcc_output = gcc_output.decode('utf-8')
            print('gcc version: {}'.format(gcc_output))
            gcc_info = gcc_output.split('.')
            gcc_major_version = int(gcc_info[0])
            gcc_minor_version = int(gcc_info[1])
        except subprocess.CalledProcessException as e:
            print('Problem getting gcc info: {}'.format(e))
            gcc_major_version = 0
            gcc_minor_version = 0
        return (gcc_major_version, gcc_minor_version)

    def parse_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up argument parser, and parse CLI args.'
        arg_parser = argparse.ArgumentParser(description='Parse the arguments for the TensorFlow build environment  setter')
        arg_parser.add_argument('--disable-mkl', dest='disable_mkl', help='Turn off MKL. By default the compiler flag --config=mkl is enabled.', action='store_true')
        arg_parser.add_argument('--disable-v2', dest='disable_v2', help='Build TensorFlow v1 rather than v2. By default the  compiler flag --config=v2 is enabled.', action='store_true')
        arg_parser.add_argument('--enable-bfloat16', dest='enable_bfloat16', help='Enable bfloat16 build. By default it is  disabled if no parameter is passed.', action='store_true')
        arg_parser.add_argument('--enable-dnnl1', dest='enable_dnnl1', help='Enable dnnl1 build. By default it is  disabled if no parameter is passed.', action='store_true')
        arg_parser.add_argument('-s', '--secure-build', dest='secure_build', help='Enable secure build flags.', action='store_true')
        arg_parser.add_argument('-p', '--platform', choices=self.PLATFORMS_.keys(), help='The target platform.', dest='target_platform', default=self.default_platform_)
        arg_parser.add_argument('-f', '--bazelrc-file', dest='bazelrc_file', help='The full path to the bazelrc file into which the build command will be written. The path will be relative to the container  environment.', required=True)
        self.args = arg_parser.parse_args()

    def validate_args(self):
        if False:
            return 10
        if os.path.exists(self.args.bazelrc_file):
            if os.path.isfile(self.args.bazelrc_file):
                self._debug('The file {} exists and will be deleted.'.format(self.args.bazelrc_file))
            elif os.path.isdir(self.args.bazelrc_file):
                print('You can\'t write bazel config to "{}" because it is a directory'.format(self.args.bazelrc_file))
                return False
        (gcc_major_version, gcc_minor_version) = self.get_gcc_version()
        if gcc_major_version == 0 or not self.target_platform_.set_host_gcc_version(gcc_major_version, gcc_minor_version):
            return False
        return True

    def set_build_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Generate Bazel build flags.'
        for flag in BASIC_BUILD_OPTS:
            self.bazel_flags_ += '{} '.format(flag)
        if self.args.secure_build:
            for flag in SECURE_BUILD_OPTS:
                self.bazel_flags_ += '{} '.format(flag)
        if not self.args.disable_mkl:
            self.bazel_flags_ += '--config=mkl '
        if self.args.disable_v2:
            self.bazel_flags_ += '--config=v1 '
        if self.args.enable_dnnl1:
            self.bazel_flags_ += '--define build_with_mkl_dnn_v1_only=true '
        if self.args.enable_bfloat16:
            self.bazel_flags_ += '--copt=-DENABLE_INTEL_MKL_BFLOAT16 '
        self.bazel_flags_ += self.target_platform_.get_bazel_gcc_flags()

    def write_build_args(self):
        if False:
            return 10
        self._debug('Writing build flags: {}'.format(self.bazel_flags_))
        with open(self.args.bazelrc_file, 'w') as f:
            f.write(self.bazel_flags_ + '\n')

    def _debug(self, msg):
        if False:
            return 10
        print(msg)

    def go(self):
        if False:
            i = 10
            return i + 15
        self.parse_args()
        self.target_platform_ = self.PLATFORMS_.get(self.args.target_platform)
        if self.validate_args():
            self.set_build_args()
            self.write_build_args()
        else:
            print('Error.')
env_setter = BuildEnvSetter()
env_setter.go()