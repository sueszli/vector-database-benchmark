import os
import site
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):

    def build_extensions(self):
        if False:
            print('Hello World!')
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()
paddle_extra_compile_args = ['-std=c++14', '-shared', '-fPIC', '-Wno-parentheses', '-DPADDLE_WITH_CUSTOM_KERNEL']
site_packages_path = site.getsitepackages()
paddle_custom_kernel_include = [os.path.join(path, 'paddle', 'include') for path in site_packages_path]
compile_third_party_path = os.path.join(os.environ['PADDLE_BINARY_DIR'], 'third_party')
paddle_custom_kernel_include += [os.path.join(compile_third_party_path, 'install/gflags/include'), os.path.join(compile_third_party_path, 'install/glog/include')]
paddle_custom_kernel_library_dir = [os.path.join(path, 'paddle', 'base') for path in site_packages_path]
libs = [':libpaddle.so']
custom_kernel_dot_module = Extension('custom_kernel_dot', sources=['custom_kernel_dot.cc'], include_dirs=paddle_custom_kernel_include, library_dirs=paddle_custom_kernel_library_dir, libraries=libs, extra_compile_args=paddle_extra_compile_args)
setup(name='custom_kernel_dot', version='1.0', description='custom kernel fot compiling', cmdclass={'build_ext': BuildExt}, ext_modules=[custom_kernel_dot_module])