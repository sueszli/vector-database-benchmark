"""SCons.Tool.javac

Tool-specific initialization for javac.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
import os
import os.path
from collections import OrderedDict
import SCons.Action
import SCons.Builder
from SCons.Node.FS import _my_normcase
from SCons.Tool.JavaCommon import parse_java_file, get_java_install_dirs, get_java_include_paths
import SCons.Util

def classname(path):
    if False:
        while True:
            i = 10
    'Turn a string (path name) into a Java class name.'
    return os.path.normpath(path).replace(os.sep, '.')

def emit_java_classes(target, source, env):
    if False:
        return 10
    'Create and return lists of source java files\n    and their corresponding target class files.\n    '
    java_suffix = env.get('JAVASUFFIX', '.java')
    class_suffix = env.get('JAVACLASSSUFFIX', '.class')
    target[0].must_be_same(SCons.Node.FS.Dir)
    classdir = target[0]
    s = source[0].rentry().disambiguate()
    if isinstance(s, SCons.Node.FS.File):
        sourcedir = s.dir.rdir()
    elif isinstance(s, SCons.Node.FS.Dir):
        sourcedir = s.rdir()
    else:
        raise SCons.Errors.UserError("Java source must be File or Dir, not '%s'" % s.__class__)
    slist = []
    js = _my_normcase(java_suffix)
    for entry in source:
        entry = entry.rentry().disambiguate()
        if isinstance(entry, SCons.Node.FS.File):
            slist.append(entry)
        elif isinstance(entry, SCons.Node.FS.Dir):
            result = OrderedDict()
            dirnode = entry.rdir()

            def find_java_files(arg, dirpath, filenames):
                if False:
                    print('Hello World!')
                java_files = sorted([n for n in filenames if _my_normcase(n).endswith(js)])
                mydir = dirnode.Dir(dirpath)
                java_paths = [mydir.File(f) for f in java_files]
                for jp in java_paths:
                    arg[jp] = True
            for (dirpath, dirnames, filenames) in os.walk(dirnode.get_abspath()):
                find_java_files(result, dirpath, filenames)
            entry.walk(find_java_files, result)
            slist.extend(list(result.keys()))
        else:
            raise SCons.Errors.UserError("Java source must be File or Dir, not '%s'" % entry.__class__)
    version = env.get('JAVAVERSION', '1.4')
    full_tlist = []
    for f in slist:
        tlist = []
        source_file_based = True
        pkg_dir = None
        if not f.is_derived():
            (pkg_dir, classes) = parse_java_file(f.rfile().get_abspath(), version)
            if classes:
                source_file_based = False
                if pkg_dir:
                    d = target[0].Dir(pkg_dir)
                    p = pkg_dir + os.sep
                else:
                    d = target[0]
                    p = ''
                for c in classes:
                    t = d.File(c + class_suffix)
                    t.attributes.java_classdir = classdir
                    t.attributes.java_sourcedir = sourcedir
                    t.attributes.java_classname = classname(p + c)
                    tlist.append(t)
        if source_file_based:
            base = f.name[:-len(java_suffix)]
            if pkg_dir:
                t = target[0].Dir(pkg_dir).File(base + class_suffix)
            else:
                t = target[0].File(base + class_suffix)
            t.attributes.java_classdir = classdir
            t.attributes.java_sourcedir = f.dir
            t.attributes.java_classname = classname(base)
            tlist.append(t)
        for t in tlist:
            t.set_specific_source([f])
        full_tlist.extend(tlist)
    return (full_tlist, slist)
JavaAction = SCons.Action.Action('$JAVACCOM', '$JAVACCOMSTR')
JavaBuilder = SCons.Builder.Builder(action=JavaAction, emitter=emit_java_classes, target_factory=SCons.Node.FS.Entry, source_factory=SCons.Node.FS.Entry)

class pathopt:
    """
    Callable object for generating javac-style path options from
    a construction variable (e.g. -classpath, -sourcepath).
    """

    def __init__(self, opt, var, default=None):
        if False:
            for i in range(10):
                print('nop')
        self.opt = opt
        self.var = var
        self.default = default

    def __call__(self, target, source, env, for_signature):
        if False:
            print('Hello World!')
        path = env[self.var]
        if path and (not SCons.Util.is_List(path)):
            path = [path]
        if self.default:
            default = env[self.default]
            if default:
                if not SCons.Util.is_List(default):
                    default = [default]
                path = path + default
        if path:
            path = SCons.Util.flatten(path)
            return [self.opt, os.pathsep.join(map(str, path))]
        else:
            return []

def Java(env, target, source, *args, **kw):
    if False:
        i = 10
        return i + 15
    '\n    A pseudo-Builder wrapper around the separate JavaClass{File,Dir}\n    Builders.\n    '
    if not SCons.Util.is_List(target):
        target = [target]
    if not SCons.Util.is_List(source):
        source = [source]
    target = target + [target[-1]] * (len(source) - len(target))
    java_suffix = env.subst('$JAVASUFFIX')
    result = []
    for (t, s) in zip(target, source):
        if isinstance(s, SCons.Node.FS.Base):
            if isinstance(s, SCons.Node.FS.File):
                b = env.JavaClassFile
            else:
                b = env.JavaClassDir
        elif os.path.isfile(s):
            b = env.JavaClassFile
        elif os.path.isdir(s):
            b = env.JavaClassDir
        elif s[-len(java_suffix):] == java_suffix:
            b = env.JavaClassFile
        else:
            b = env.JavaClassDir
        result.extend(b(t, s, *args, **kw))
    return result

def generate(env):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for javac to an Environment.'
    java_file = SCons.Tool.CreateJavaFileBuilder(env)
    java_class = SCons.Tool.CreateJavaClassFileBuilder(env)
    java_class_dir = SCons.Tool.CreateJavaClassDirBuilder(env)
    java_class.add_emitter(None, emit_java_classes)
    java_class.add_emitter(env.subst('$JAVASUFFIX'), emit_java_classes)
    java_class_dir.emitter = emit_java_classes
    env.AddMethod(Java)
    version = env.get('JAVAVERSION', None)
    if env['PLATFORM'] == 'win32':
        paths = get_java_install_dirs('win32', version=version)
        javac = SCons.Tool.find_program_path(env, 'javac', default_paths=paths)
        if javac:
            javac_bin_dir = os.path.dirname(javac)
            env.AppendENVPath('PATH', javac_bin_dir)
    else:
        javac = SCons.Tool.find_program_path(env, 'javac')
    env.SetDefault(JAVAC='javac', JAVACFLAGS=SCons.Util.CLVar(''), JAVAINCLUDES=get_java_include_paths(env, javac, version), JAVACLASSSUFFIX='.class', JAVASUFFIX='.java', JAVABOOTCLASSPATH=[], JAVACLASSPATH=[], JAVASOURCEPATH=[], JAVAPROCESSORPATH=[])
    env['_javapathopt'] = pathopt
    env['_JAVABOOTCLASSPATH'] = '${_javapathopt("-bootclasspath", "JAVABOOTCLASSPATH")} '
    env['_JAVAPROCESSORPATH'] = '${_javapathopt("-processorpath", "JAVAPROCESSORPATH")} '
    env['_JAVACLASSPATH'] = '${_javapathopt("-classpath", "JAVACLASSPATH")} '
    env['_JAVASOURCEPATH'] = '${_javapathopt("-sourcepath", "JAVASOURCEPATH", "_JAVASOURCEPATHDEFAULT")} '
    env['_JAVASOURCEPATHDEFAULT'] = '${TARGET.attributes.java_sourcedir}'
    env['_JAVACCOM'] = '$JAVAC $JAVACFLAGS $_JAVABOOTCLASSPATH $_JAVAPROCESSORPATH $_JAVACLASSPATH -d ${TARGET.attributes.java_classdir} $_JAVASOURCEPATH $SOURCES'
    env['JAVACCOM'] = "${TEMPFILE('$_JAVACCOM','$JAVACCOMSTR')}"

def exists(env):
    if False:
        for i in range(10):
            print('nop')
    return 1