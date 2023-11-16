"""Helpers for OpenMP support during the build."""
import os
import sys
import textwrap
import warnings
from .pre_build_helpers import compile_test_program

def get_openmp_flag():
    if False:
        while True:
            i = 10
    if sys.platform == 'win32':
        return ['/openmp']
    elif sys.platform == 'darwin' and 'openmp' in os.getenv('CPPFLAGS', ''):
        return []
    return ['-fopenmp']

def check_openmp_support():
    if False:
        i = 10
        return i + 15
    'Check whether OpenMP test code can be compiled and run'
    if 'PYODIDE_PACKAGE_ABI' in os.environ:
        return False
    code = textwrap.dedent('        #include <omp.h>\n        #include <stdio.h>\n        int main(void) {\n        #pragma omp parallel\n        printf("nthreads=%d\\n", omp_get_num_threads());\n        return 0;\n        }\n        ')
    extra_preargs = os.getenv('LDFLAGS', None)
    if extra_preargs is not None:
        extra_preargs = extra_preargs.strip().split(' ')
        extra_preargs = [flag for flag in extra_preargs if flag.startswith(('-L', '-Wl,-rpath', '-l', '-Wl,--sysroot=/'))]
    extra_postargs = get_openmp_flag()
    openmp_exception = None
    try:
        output = compile_test_program(code, extra_preargs=extra_preargs, extra_postargs=extra_postargs)
        if output and 'nthreads=' in output[0]:
            nthreads = int(output[0].strip().split('=')[1])
            openmp_supported = len(output) == nthreads
        elif 'PYTHON_CROSSENV' in os.environ:
            openmp_supported = True
        else:
            openmp_supported = False
    except Exception as exception:
        openmp_supported = False
        openmp_exception = exception
    if not openmp_supported:
        if os.getenv('SKLEARN_FAIL_NO_OPENMP'):
            raise Exception('Failed to build scikit-learn with OpenMP support') from openmp_exception
        else:
            message = textwrap.dedent('\n\n                                ***********\n                                * WARNING *\n                                ***********\n\n                It seems that scikit-learn cannot be built with OpenMP.\n\n                - Make sure you have followed the installation instructions:\n\n                    https://scikit-learn.org/dev/developers/advanced_installation.html\n\n                - If your compiler supports OpenMP but you still see this\n                  message, please submit a bug report at:\n\n                    https://github.com/scikit-learn/scikit-learn/issues\n\n                - The build will continue with OpenMP-based parallelism\n                  disabled. Note however that some estimators will run in\n                  sequential mode instead of leveraging thread-based\n                  parallelism.\n\n                                    ***\n                ')
            warnings.warn(message)
    return openmp_supported