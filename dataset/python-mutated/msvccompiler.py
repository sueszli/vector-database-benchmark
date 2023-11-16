import os
from distutils.msvccompiler import MSVCCompiler as _MSVCCompiler
from .system_info import platform_bits

def _merge(old, new):
    if False:
        return 10
    'Concatenate two environment paths avoiding repeats.\n\n    Here `old` is the environment string before the base class initialize\n    function is called and `new` is the string after the call. The new string\n    will be a fixed string if it is not obtained from the current environment,\n    or the same as the old string if obtained from the same environment. The aim\n    here is not to append the new string if it is already contained in the old\n    string so as to limit the growth of the environment string.\n\n    Parameters\n    ----------\n    old : string\n        Previous environment string.\n    new : string\n        New environment string.\n\n    Returns\n    -------\n    ret : string\n        Updated environment string.\n\n    '
    if new in old:
        return old
    if not old:
        return new
    return ';'.join([old, new])

class MSVCCompiler(_MSVCCompiler):

    def __init__(self, verbose=0, dry_run=0, force=0):
        if False:
            print('Hello World!')
        _MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self):
        if False:
            return 10
        environ_lib = os.getenv('lib', '')
        environ_include = os.getenv('include', '')
        _MSVCCompiler.initialize(self)
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])
        os.environ['include'] = _merge(environ_include, os.environ['include'])
        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']

def lib_opts_if_msvc(build_cmd):
    if False:
        while True:
            i = 10
    " Add flags if we are using MSVC compiler\n\n    We can't see `build_cmd` in our scope, because we have not initialized\n    the distutils build command, so use this deferred calculation to run\n    when we are building the library.\n    "
    if build_cmd.compiler.compiler_type != 'msvc':
        return []
    flags = ['/GL-']
    if build_cmd.compiler_opt.cc_test_flags(['-d2VolatileMetadata-']):
        flags.append('-d2VolatileMetadata-')
    return flags