import os
import SCons.Node
import SCons.Node.FS
import SCons.Scanner
from SCons.Util import flatten, is_String

def _subst_paths(env, paths) -> list:
    if False:
        return 10
    'Return a list of substituted path elements.\n\n    If *paths* is a string, it is split on the search-path separator.\n    Otherwise, substitution is done on string-valued list elements but\n    they are not split.\n\n    Note helps support behavior like pulling in the external ``CLASSPATH``\n    and setting it directly into ``JAVACLASSPATH``, however splitting on\n    ``os.pathsep`` makes the interpretation system-specific (this is\n    warned about in the manpage entry for ``JAVACLASSPATH``).\n    '
    if is_String(paths):
        paths = env.subst(paths)
        if SCons.Util.is_String(paths):
            paths = paths.split(os.pathsep)
    else:
        paths = flatten(paths)
        paths = [env.subst(path) if is_String(path) else path for path in paths]
    return paths

def _collect_classes(classlist, dirname, files):
    if False:
        return 10
    for fname in files:
        if fname.endswith('.class'):
            classlist.append(os.path.join(str(dirname), fname))

def scan(node, env, libpath=()) -> list:
    if False:
        print('Hello World!')
    'Scan for files both on JAVACLASSPATH and JAVAPROCESSORPATH.\n\n    JAVACLASSPATH/JAVAPROCESSORPATH path can contain:\n     - Explicit paths to JAR/Zip files\n     - Wildcards (*)\n     - Directories which contain classes in an unnamed package\n     - Parent directories of the root package for classes in a named package\n\n    Class path entries that are neither directories nor archives (.zip\n    or JAR files) nor the asterisk (*) wildcard character are ignored.\n    '
    classpath = []
    for var in ['JAVACLASSPATH', 'JAVAPROCESSORPATH']:
        classpath += _subst_paths(env, env.get(var, []))
    result = []
    for path in classpath:
        if is_String(path) and '*' in path:
            libs = env.Glob(path)
        else:
            libs = [path]
        for lib in libs:
            if os.path.isdir(str(lib)):
                env.Dir(lib).walk(_collect_classes, result)
                for (root, dirs, files) in os.walk(str(lib)):
                    _collect_classes(result, root, files)
            else:
                result.append(lib)
    return list(filter(lambda x: os.path.splitext(str(x))[1] in ['.class', '.zip', '.jar'], result))

def JavaScanner():
    if False:
        while True:
            i = 10
    'Scanner for .java files.\n\n    .. versionadded:: 4.4\n    '
    return SCons.Scanner.Base(scan, 'JavaScanner', skeys=['.java'])