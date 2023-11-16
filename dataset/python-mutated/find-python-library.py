import sys, sysconfig, os, distutils.sysconfig as du_sysconfig, itertools

def get_python_library(python_version):
    if False:
        for i in range(10):
            print('nop')
    'Get path to the python library associated with the current python\n    interpreter. Adapted from https://github.com/scikit-build/scikit-build/blob/master/skbuild/cmaker.py#L272'
    python_library = sysconfig.get_config_var('LIBRARY')
    if not python_library or os.path.splitext(python_library)[1][-2:] == '.a':
        candidate_lib_prefixes = ['', 'lib']
        candidate_implementations = ['python']
        if hasattr(sys, 'pypy_version_info'):
            candidate_implementations = ['pypy-c', 'pypy3-c']
        candidate_extensions = ['.lib', '.so', '.a']
        if sysconfig.get_config_var('WITH_DYLD'):
            candidate_extensions.insert(0, '.dylib')
        candidate_versions = [python_version]
        if python_version:
            candidate_versions.append('')
            candidate_versions.insert(0, ''.join(python_version.split('.')[:2]))
        abiflags = getattr(sys, 'abiflags', '')
        candidate_abiflags = [abiflags]
        if abiflags:
            candidate_abiflags.append('')
        libdir = du_sysconfig.get_config_var('LIBDIR')
        if sysconfig.get_config_var('MULTIARCH'):
            masd = sysconfig.get_config_var('multiarchsubdir')
            if masd:
                if masd.startswith(os.sep):
                    masd = masd[len(os.sep):]
                libdir = os.path.join(libdir, masd)
        if libdir is None:
            libdir = os.path.abspath(os.path.join(sysconfig.get_config_var('LIBDEST'), '..', 'libs'))
        candidates = [os.path.join(sysconfig.get_config_var('LIBPL'), sysconfig.get_config_var('LDLIBRARY'))] + [os.path.join(libdir, ''.join((pre, impl, ver, abi, ext))) for (pre, impl, ext, ver, abi) in itertools.product(candidate_lib_prefixes, candidate_implementations, candidate_extensions, candidate_versions, candidate_abiflags)]
        for candidate in candidates:
            if os.path.exists(candidate):
                python_library = candidate
                break
    return python_library
print(get_python_library('{}.{}'.format(sys.version_info[0], sys.version_info[1])))