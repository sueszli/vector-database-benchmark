"""SCons.Tool.rmic

Tool-specific initialization for rmic.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/rmic.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os.path
import SCons.Action
import SCons.Builder
import SCons.Node.FS
import SCons.Util
from SCons.Tool.JavaCommon import get_java_install_dirs

def emit_rmic_classes(target, source, env):
    if False:
        while True:
            i = 10
    'Create and return lists of Java RMI stub and skeleton\n    class files to be created from a set of class files.\n    '
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
            if class_suffix and classname[:-len(class_suffix)] == class_suffix:
                classname = classname[-len(class_suffix):]
        s = src.rfile()
        s.attributes.java_classdir = classdir
        s.attributes.java_classname = classname
        slist.append(s)
    stub_suffixes = ['_Stub']
    if env.get('JAVAVERSION') == '1.4':
        stub_suffixes.append('_Skel')
    tlist = []
    for s in source:
        for suff in stub_suffixes:
            fname = s.attributes.java_classname.replace('.', os.sep) + suff + class_suffix
            t = target[0].File(fname)
            t.attributes.java_lookupdir = target[0]
            tlist.append(t)
    return (tlist, source)
RMICAction = SCons.Action.Action('$RMICCOM', '$RMICCOMSTR')
RMICBuilder = SCons.Builder.Builder(action=RMICAction, emitter=emit_rmic_classes, src_suffix='$JAVACLASSSUFFIX', target_factory=SCons.Node.FS.Dir, source_factory=SCons.Node.FS.File)

def generate(env):
    if False:
        i = 10
        return i + 15
    'Add Builders and construction variables for rmic to an Environment.'
    env['BUILDERS']['RMIC'] = RMICBuilder
    if env['PLATFORM'] == 'win32':
        version = env.get('JAVAVERSION', None)
        paths = get_java_install_dirs('win32', version=version)
        rmic = SCons.Tool.find_program_path(env, 'rmic', default_paths=paths)
        if rmic:
            rmic_bin_dir = os.path.dirname(rmic)
            env.AppendENVPath('PATH', rmic_bin_dir)
    env['RMIC'] = 'rmic'
    env['RMICFLAGS'] = SCons.Util.CLVar('')
    env['RMICCOM'] = '$RMIC $RMICFLAGS -d ${TARGET.attributes.java_lookupdir} -classpath ${SOURCE.attributes.java_classdir} ${SOURCES.attributes.java_classname}'
    env['JAVACLASSSUFFIX'] = '.class'

def exists(env):
    if False:
        while True:
            i = 10
    return 1