__revision__ = 'src/engine/SCons/Scanner/Prog.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Node
import SCons.Node.FS
import SCons.Scanner
import SCons.Util
print_find_libs = None

def ProgramScanner(**kw):
    if False:
        for i in range(10):
            print('nop')
    'Return a prototype Scanner instance for scanning executable\n    files for static-lib dependencies'
    kw['path_function'] = SCons.Scanner.FindPathDirs('LIBPATH')
    ps = SCons.Scanner.Base(scan, 'ProgramScanner', **kw)
    return ps

def _subst_libs(env, libs):
    if False:
        return 10
    '\n    Substitute environment variables and split into list.\n    '
    if SCons.Util.is_String(libs):
        libs = env.subst(libs)
        if SCons.Util.is_String(libs):
            libs = libs.split()
    elif SCons.Util.is_Sequence(libs):
        _libs = []
        for l in libs:
            _libs += _subst_libs(env, l)
        libs = _libs
    else:
        libs = [libs]
    return libs

def scan(node, env, libpath=()):
    if False:
        for i in range(10):
            print('nop')
    '\n    This scanner scans program files for static-library\n    dependencies.  It will search the LIBPATH environment variable\n    for libraries specified in the LIBS variable, returning any\n    files it finds as dependencies.\n    '
    try:
        libs = env['LIBS']
    except KeyError:
        return []
    libs = _subst_libs(env, libs)
    try:
        prefix = env['LIBPREFIXES']
        if not SCons.Util.is_List(prefix):
            prefix = [prefix]
    except KeyError:
        prefix = ['']
    try:
        suffix = env['LIBSUFFIXES']
        if not SCons.Util.is_List(suffix):
            suffix = [suffix]
    except KeyError:
        suffix = ['']
    pairs = []
    for suf in map(env.subst, suffix):
        for pref in map(env.subst, prefix):
            pairs.append((pref, suf))
    result = []
    if callable(libpath):
        libpath = libpath()
    find_file = SCons.Node.FS.find_file
    adjustixes = SCons.Util.adjustixes
    for lib in libs:
        if SCons.Util.is_String(lib):
            for (pref, suf) in pairs:
                l = adjustixes(lib, pref, suf)
                l = find_file(l, libpath, verbose=print_find_libs)
                if l:
                    result.append(l)
        else:
            result.append(lib)
    return result