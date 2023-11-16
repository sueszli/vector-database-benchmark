import os

def get_include_dir():
    if False:
        while True:
            i = 10
    'Get the path to the directory containing C++ header files.\n\n    Returns:\n        String representing the path to the include directory\n    '
    import nvidia.dali as dali
    return os.path.join(os.path.dirname(dali.__file__), 'include')

def get_lib_dir():
    if False:
        print('Hello World!')
    'Get the path to the directory containing DALI library.\n\n    Returns:\n        String representing the path to the library directory\n    '
    import nvidia.dali as dali
    return os.path.dirname(dali.__file__)

def get_include_flags():
    if False:
        while True:
            i = 10
    'Get the include flags for custom operators\n\n    Returns:\n        The compilation flags\n    '
    flags = []
    flags.append('-I%s' % get_include_dir())
    return flags

def get_compile_flags():
    if False:
        print('Hello World!')
    'Get the compilation flags for custom operators\n\n    Returns:\n        The compilation flags\n    '
    import nvidia.dali.backend as b
    flags = []
    flags.append('-I%s' % get_include_dir())
    flags.append('-D_GLIBCXX_USE_CXX11_ABI=%d' % b.GetCxx11AbiFlag())
    return flags

def get_link_flags():
    if False:
        print('Hello World!')
    'Get the link flags for custom operators\n\n    Returns:\n        The link flags\n    '
    flags = []
    flags.append('-L%s' % get_lib_dir())
    flags.append('-ldali')
    return flags