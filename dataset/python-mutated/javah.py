"""SCons.Tool.javah

Tool-specific initialization for javah.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
import os.path
import SCons.Action
import SCons.Builder
import SCons.Node.FS
import SCons.Tool.javac
import SCons.Util
from SCons.Tool.JavaCommon import get_java_install_dirs

def emit_java_headers(target, source, env):
    if False:
        print('Hello World!')
    'Create and return lists of Java stub header files that will\n    be created from a set of class files.\n    '
    class_suffix = env.get('JAVACLASSSUFFIX', '.class')
    classdir = env.get('JAVACLASSDIR')
    if not classdir:
        try:
            s = source[0]
        except IndexError:
            classdir = '.'
        else:
            try:
                classdir = s.attributes.java_classdir
            except AttributeError:
                classdir = '.'
    classdir = env.Dir(classdir).rdir()
    if str(classdir) == '.':
        c_ = None
    else:
        c_ = str(classdir) + os.sep
    slist = []
    for src in source:
        try:
            classname = src.attributes.java_classname
        except AttributeError:
            classname = str(src)
            if c_ and classname[:len(c_)] == c_:
                classname = classname[len(c_):]
            if class_suffix and classname[-len(class_suffix):] == class_suffix:
                classname = classname[:-len(class_suffix)]
            classname = SCons.Tool.javac.classname(classname)
        s = src.rfile()
        s.attributes.java_classname = classname
        slist.append(s)
    s = source[0].rfile()
    if not hasattr(s.attributes, 'java_classdir'):
        s.attributes.java_classdir = classdir
    if target[0].__class__ is SCons.Node.FS.File:
        tlist = target
    else:
        if not isinstance(target[0], SCons.Node.FS.Dir):
            target[0].__class__ = SCons.Node.FS.Dir
            target[0]._morph()
        tlist = []
        for s in source:
            fname = s.attributes.java_classname.replace('.', '_') + '.h'
            t = target[0].File(fname)
            t.attributes.java_lookupdir = target[0]
            tlist.append(t)
    return (tlist, source)

def JavaHOutFlagGenerator(target, source, env, for_signature):
    if False:
        return 10
    try:
        t = target[0]
    except (AttributeError, IndexError, TypeError):
        t = target
    try:
        return '-d ' + str(t.attributes.java_lookupdir)
    except AttributeError:
        return '-o ' + str(t)

def getJavaHClassPath(env, target, source, for_signature):
    if False:
        for i in range(10):
            print('nop')
    path = '${SOURCE.attributes.java_classdir}'
    if 'JAVACLASSPATH' in env and env['JAVACLASSPATH']:
        path = SCons.Util.AppendPath(path, env['JAVACLASSPATH'])
    return '-classpath %s' % path

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for javah to an Environment.'
    java_javah = SCons.Tool.CreateJavaHBuilder(env)
    java_javah.emitter = emit_java_headers
    if env['PLATFORM'] == 'win32':
        paths = get_java_install_dirs('win32')
        javah = SCons.Tool.find_program_path(env, 'javah', default_paths=paths)
        if javah:
            javah_bin_dir = os.path.dirname(javah)
            env.AppendENVPath('PATH', javah_bin_dir)
    env.SetDefault(JAVAH='javah', JAVAHFLAGS=SCons.Util.CLVar(''), JAVACLASSSUFFIX='.class', JAVASUFFIX='.java')
    env['_JAVAHOUTFLAG'] = JavaHOutFlagGenerator
    env['_JAVAHCLASSPATH'] = getJavaHClassPath
    env['JAVAHCOM'] = '$JAVAH $JAVAHFLAGS $_JAVAHOUTFLAG $_JAVAHCLASSPATH ${SOURCES.attributes.java_classname}'

def exists(env):
    if False:
        return 10
    return env.Detect('javah')