import os
from distutils.msvc9compiler import MSVCCompiler as _MSVCCompiler
from .system_info import platform_bits

def _merge(old, new):
    if False:
        i = 10
        return i + 15
    'Concatenate two environment paths avoiding repeats.\n\n    Here `old` is the environment string before the base class initialize\n    function is called and `new` is the string after the call. The new string\n    will be a fixed string if it is not obtained from the current environment,\n    or the same as the old string if obtained from the same environment. The aim\n    here is not to append the new string if it is already contained in the old\n    string so as to limit the growth of the environment string.\n\n    Parameters\n    ----------\n    old : string\n        Previous environment string.\n    new : string\n        New environment string.\n\n    Returns\n    -------\n    ret : string\n        Updated environment string.\n\n    '
    if not old:
        return new
    if new in old:
        return old
    return ';'.join([old, new])

class MSVCCompiler(_MSVCCompiler):

    def __init__(self, verbose=0, dry_run=0, force=0):
        if False:
            print('Hello World!')
        _MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self, plat_name=None):
        if False:
            print('Hello World!')
        environ_lib = os.getenv('lib')
        environ_include = os.getenv('include')
        _MSVCCompiler.initialize(self, plat_name)
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])
        os.environ['include'] = _merge(environ_include, os.environ['include'])
        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']

    def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
        if False:
            print('Hello World!')
        ld_args.append('/MANIFEST')
        _MSVCCompiler.manifest_setup_ldargs(self, output_filename, build_temp, ld_args)