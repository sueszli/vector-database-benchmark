import os
import pytest
import spack.build_environment
import spack.repo
import spack.spec
from spack.package import build_system_flags, env_flags, inject_flags

@pytest.fixture()
def temp_env():
    if False:
        i = 10
        return i + 15
    old_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(old_env)

def add_o3_to_build_system_cflags(pkg, name, flags):
    if False:
        while True:
            i = 10
    build_system_flags = []
    if name == 'cflags':
        build_system_flags.append('-O3')
    return (flags, None, build_system_flags)

@pytest.mark.usefixtures('config', 'mock_packages')
class TestFlagHandlers:

    def test_no_build_system_flags(self, temp_env):
        if False:
            print('Hello World!')
        s1 = spack.spec.Spec('cmake-client').concretized()
        spack.build_environment.setup_package(s1.package, False)
        s2 = spack.spec.Spec('patchelf').concretized()
        spack.build_environment.setup_package(s2.package, False)
        assert 'SPACK_CPPFLAGS' not in os.environ
        assert 'CPPFLAGS' not in os.environ

    def test_unbound_method(self, temp_env):
        if False:
            print('Hello World!')
        s = spack.spec.Spec('mpileaks cppflags=-g').concretized()
        s.package.flag_handler = s.package.__class__.inject_flags
        spack.build_environment.setup_package(s.package, False)
        assert os.environ['SPACK_CPPFLAGS'] == '-g'
        assert 'CPPFLAGS' not in os.environ

    def test_inject_flags(self, temp_env):
        if False:
            while True:
                i = 10
        s = spack.spec.Spec('mpileaks cppflags=-g').concretized()
        s.package.flag_handler = inject_flags
        spack.build_environment.setup_package(s.package, False)
        assert os.environ['SPACK_CPPFLAGS'] == '-g'
        assert 'CPPFLAGS' not in os.environ

    def test_env_flags(self, temp_env):
        if False:
            return 10
        s = spack.spec.Spec('mpileaks cppflags=-g').concretized()
        s.package.flag_handler = env_flags
        spack.build_environment.setup_package(s.package, False)
        assert os.environ['CPPFLAGS'] == '-g'
        assert 'SPACK_CPPFLAGS' not in os.environ

    def test_build_system_flags_cmake(self, temp_env):
        if False:
            i = 10
            return i + 15
        s = spack.spec.Spec('cmake-client cppflags=-g').concretized()
        s.package.flag_handler = build_system_flags
        spack.build_environment.setup_package(s.package, False)
        assert 'SPACK_CPPFLAGS' not in os.environ
        assert 'CPPFLAGS' not in os.environ
        assert set(s.package.cmake_flag_args) == {'-DCMAKE_C_FLAGS=-g', '-DCMAKE_CXX_FLAGS=-g', '-DCMAKE_Fortran_FLAGS=-g'}

    def test_build_system_flags_autotools(self, temp_env):
        if False:
            print('Hello World!')
        s = spack.spec.Spec('patchelf cppflags=-g').concretized()
        s.package.flag_handler = build_system_flags
        spack.build_environment.setup_package(s.package, False)
        assert 'SPACK_CPPFLAGS' not in os.environ
        assert 'CPPFLAGS' not in os.environ
        assert 'CPPFLAGS=-g' in s.package.configure_flag_args

    def test_build_system_flags_not_implemented(self, temp_env):
        if False:
            print('Hello World!')
        'Test the command line flags method raises a NotImplementedError'
        s = spack.spec.Spec('mpileaks cppflags=-g').concretized()
        s.package.flag_handler = build_system_flags
        try:
            spack.build_environment.setup_package(s.package, False)
            assert False
        except NotImplementedError:
            assert True

    def test_add_build_system_flags_autotools(self, temp_env):
        if False:
            for i in range(10):
                print('nop')
        s = spack.spec.Spec('patchelf cppflags=-g').concretized()
        s.package.flag_handler = add_o3_to_build_system_cflags
        spack.build_environment.setup_package(s.package, False)
        assert '-g' in os.environ['SPACK_CPPFLAGS']
        assert 'CPPFLAGS' not in os.environ
        assert s.package.configure_flag_args == ['CFLAGS=-O3']

    def test_add_build_system_flags_cmake(self, temp_env):
        if False:
            while True:
                i = 10
        s = spack.spec.Spec('cmake-client cppflags=-g').concretized()
        s.package.flag_handler = add_o3_to_build_system_cflags
        spack.build_environment.setup_package(s.package, False)
        assert '-g' in os.environ['SPACK_CPPFLAGS']
        assert 'CPPFLAGS' not in os.environ
        assert s.package.cmake_flag_args == ['-DCMAKE_C_FLAGS=-O3']

    def test_ld_flags_cmake(self, temp_env):
        if False:
            for i in range(10):
                print('nop')
        s = spack.spec.Spec('cmake-client ldflags=-mthreads').concretized()
        s.package.flag_handler = build_system_flags
        spack.build_environment.setup_package(s.package, False)
        assert 'SPACK_LDFLAGS' not in os.environ
        assert 'LDFLAGS' not in os.environ
        assert set(s.package.cmake_flag_args) == {'-DCMAKE_EXE_LINKER_FLAGS=-mthreads', '-DCMAKE_MODULE_LINKER_FLAGS=-mthreads', '-DCMAKE_SHARED_LINKER_FLAGS=-mthreads'}

    def test_ld_libs_cmake(self, temp_env):
        if False:
            for i in range(10):
                print('nop')
        s = spack.spec.Spec('cmake-client ldlibs=-lfoo').concretized()
        s.package.flag_handler = build_system_flags
        spack.build_environment.setup_package(s.package, False)
        assert 'SPACK_LDLIBS' not in os.environ
        assert 'LDLIBS' not in os.environ
        assert set(s.package.cmake_flag_args) == {'-DCMAKE_C_STANDARD_LIBRARIES=-lfoo', '-DCMAKE_CXX_STANDARD_LIBRARIES=-lfoo', '-DCMAKE_Fortran_STANDARD_LIBRARIES=-lfoo'}

    def test_flag_handler_no_modify_specs(self, temp_env):
        if False:
            for i in range(10):
                print('nop')

        def test_flag_handler(self, name, flags):
            if False:
                print('Hello World!')
            flags.append('-foo')
            return (flags, None, None)
        s = spack.spec.Spec('cmake-client').concretized()
        s.package.flag_handler = test_flag_handler
        spack.build_environment.setup_package(s.package, False)
        assert not s.compiler_flags['cflags']
        assert os.environ['SPACK_CFLAGS'] == '-foo'