import sys
import os
import re

def get_lib_dirs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Anaconda specific\n    '
    if sys.platform == 'win32':
        dirnames = ['DLLs', os.path.join('Library', 'bin')]
    else:
        dirnames = ['lib']
    libdirs = [os.path.join(sys.prefix, x) for x in dirnames]
    return libdirs
DLLNAMEMAP = {'linux': 'lib%(name)s\\.so\\.%(ver)s$', 'linux2': 'lib%(name)s\\.so\\.%(ver)s$', 'linux-static': 'lib%(name)s\\.a$', 'darwin': 'lib%(name)s\\.%(ver)s\\.dylib$', 'win32': '%(name)s%(ver)s\\.dll$', 'win32-static': '%(name)s\\.lib$', 'bsd': 'lib%(name)s\\.so\\.%(ver)s$'}
RE_VER = '[0-9]*([_\\.][0-9]+)*'

def find_lib(libname, libdir=None, platform=None, static=False):
    if False:
        i = 10
        return i + 15
    platform = platform or sys.platform
    platform = 'bsd' if 'bsd' in platform else platform
    if static:
        platform = f'{platform}-static'
    if platform not in DLLNAMEMAP:
        return []
    pat = DLLNAMEMAP[platform] % {'name': libname, 'ver': RE_VER}
    regex = re.compile(pat)
    return find_file(regex, libdir)

def find_file(pat, libdir=None):
    if False:
        i = 10
        return i + 15
    if libdir is None:
        libdirs = get_lib_dirs()
    elif isinstance(libdir, str):
        libdirs = [libdir]
    else:
        libdirs = list(libdir)
    files = []
    for ldir in libdirs:
        try:
            entries = os.listdir(ldir)
        except FileNotFoundError:
            continue
        candidates = [os.path.join(ldir, ent) for ent in entries if pat.match(ent)]
        files.extend([c for c in candidates if os.path.isfile(c)])
    return files