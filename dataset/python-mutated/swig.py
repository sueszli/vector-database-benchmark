"""Tool-specific initialization for swig.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.
"""
import os.path
import re
import subprocess
import sys
import SCons.Action
import SCons.Defaults
import SCons.Node
import SCons.Tool
import SCons.Util
import SCons.Warnings
verbose = False
swigs = ['swig', 'swig3.0', 'swig2.0']
SwigAction = SCons.Action.Action('$SWIGCOM', '$SWIGCOMSTR')

def swigSuffixEmitter(env, source):
    if False:
        print('Hello World!')
    if '-c++' in SCons.Util.CLVar(env.subst('$SWIGFLAGS', source=source)):
        return '$SWIGCXXFILESUFFIX'
    else:
        return '$SWIGCFILESUFFIX'
_reModule = re.compile('%module(\\s*\\(.*\\))?\\s+("?)(\\S+)\\2')

def _find_modules(src):
    if False:
        while True:
            i = 10
    'Find all modules referenced by %module lines in `src`, a SWIG .i file.\n       Returns a list of all modules, and a flag set if SWIG directors have\n       been requested (SWIG will generate an additional header file in this\n       case.)'
    directors = 0
    mnames = []
    try:
        with open(src) as f:
            data = f.read()
        matches = _reModule.findall(data)
    except IOError:
        matches = []
        mnames.append(os.path.splitext(os.path.basename(src))[0])
    for m in matches:
        mnames.append(m[2])
        directors = directors or 'directors' in m[0]
    return (mnames, directors)

def _add_director_header_targets(target, env):
    if False:
        return 10
    suffix = env.subst(env['SWIGCXXFILESUFFIX'])
    for x in target[:]:
        n = x.name
        d = x.dir
        if n[-len(suffix):] == suffix:
            target.append(d.File(n[:-len(suffix)] + env['SWIGDIRECTORSUFFIX']))

def _swigEmitter(target, source, env):
    if False:
        print('Hello World!')
    swigflags = env.subst('$SWIGFLAGS', target=target, source=source)
    flags = SCons.Util.CLVar(swigflags)
    for src in source:
        src = str(src.rfile())
        mnames = None
        if '-python' in flags and '-noproxy' not in flags:
            if mnames is None:
                (mnames, directors) = _find_modules(src)
            if directors:
                _add_director_header_targets(target, env)
            python_files = [m + '.py' for m in mnames]
            outdir = env.subst('$SWIGOUTDIR', target=target, source=source)
            if outdir:
                python_files = [env.fs.File(os.path.join(outdir, j)) for j in python_files]
            else:
                python_files = [target[0].dir.File(m) for m in python_files]
            target.extend(python_files)
        if '-java' in flags:
            if mnames is None:
                (mnames, directors) = _find_modules(src)
            if directors:
                _add_director_header_targets(target, env)
            java_files = [[m + '.java', m + 'JNI.java'] for m in mnames]
            java_files = SCons.Util.flatten(java_files)
            outdir = env.subst('$SWIGOUTDIR', target=target, source=source)
            if outdir:
                java_files = [os.path.join(outdir, j) for j in java_files]
            java_files = list(map(env.fs.File, java_files))

            def t_from_s(t, p, s, x):
                if False:
                    return 10
                return t.dir
            tsm = SCons.Node._target_from_source_map
            tkey = len(tsm)
            tsm[tkey] = t_from_s
            for jf in java_files:
                jf._func_target_from_source = tkey
            target.extend(java_files)
    return (target, source)

def _get_swig_version(env, swig):
    if False:
        print('Hello World!')
    'Run the SWIG command line tool to get and return the version number'
    version = None
    swig = env.subst(swig)
    if not swig:
        return version
    pipe = SCons.Action._subproc(env, SCons.Util.CLVar(swig) + ['-version'], stdin='devnull', stderr='devnull', stdout=subprocess.PIPE)
    if pipe.wait() != 0:
        return version
    with pipe.stdout:
        out = SCons.Util.to_str(pipe.stdout.read())
    match = re.search('SWIG Version\\s+(\\S+).*', out, re.MULTILINE)
    if match:
        version = match.group(1)
        if verbose:
            print('Version is: %s' % version)
    elif verbose:
        print('Unable to detect version: [%s]' % out)
    return version

def generate(env):
    if False:
        return 10
    'Add Builders and construction variables for swig to an Environment.'
    (c_file, cxx_file) = SCons.Tool.createCFileBuilders(env)
    c_file.suffix['.i'] = swigSuffixEmitter
    cxx_file.suffix['.i'] = swigSuffixEmitter
    c_file.add_action('.i', SwigAction)
    c_file.add_emitter('.i', _swigEmitter)
    cxx_file.add_action('.i', SwigAction)
    cxx_file.add_emitter('.i', _swigEmitter)
    java_file = SCons.Tool.CreateJavaFileBuilder(env)
    java_file.suffix['.i'] = swigSuffixEmitter
    java_file.add_action('.i', SwigAction)
    java_file.add_emitter('.i', _swigEmitter)
    from SCons.Platform.mingw import MINGW_DEFAULT_PATHS
    from SCons.Platform.cygwin import CYGWIN_DEFAULT_PATHS
    from SCons.Platform.win32 import CHOCO_DEFAULT_PATH
    if sys.platform == 'win32':
        swig = SCons.Tool.find_program_path(env, 'swig', default_paths=MINGW_DEFAULT_PATHS + CYGWIN_DEFAULT_PATHS + CHOCO_DEFAULT_PATH)
        if swig:
            swig_bin_dir = os.path.dirname(swig)
            env.AppendENVPath('PATH', swig_bin_dir)
        else:
            SCons.Warnings.warn(SCons.Warnings.SConsWarning, 'swig tool requested, but binary not found in ENV PATH')
    if 'SWIG' not in env:
        env['SWIG'] = env.Detect(swigs) or swigs[0]
    env['SWIGVERSION'] = _get_swig_version(env, env['SWIG'])
    env['SWIGFLAGS'] = SCons.Util.CLVar('')
    env['SWIGDIRECTORSUFFIX'] = '_wrap.h'
    env['SWIGCFILESUFFIX'] = '_wrap$CFILESUFFIX'
    env['SWIGCXXFILESUFFIX'] = '_wrap$CXXFILESUFFIX'
    env['_SWIGOUTDIR'] = '${"-outdir \\"%s\\"" % SWIGOUTDIR}'
    env['SWIGPATH'] = []
    env['SWIGINCPREFIX'] = '-I'
    env['SWIGINCSUFFIX'] = ''
    env['_SWIGINCFLAGS'] = '${_concat(SWIGINCPREFIX, SWIGPATH, SWIGINCSUFFIX,__env__, RDirs, TARGET, SOURCE, affect_signature=False)}'
    env['SWIGCOM'] = '$SWIG -o $TARGET ${_SWIGOUTDIR} ${_SWIGINCFLAGS} $SWIGFLAGS $SOURCES'

def exists(env):
    if False:
        while True:
            i = 10
    swig = env.get('SWIG') or env.Detect(['swig'])
    return swig