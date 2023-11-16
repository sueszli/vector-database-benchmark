"""SCons.Tool.ldc

Tool-specific initialization for the LDC compiler.
(https://github.com/ldc-developers/ldc)

Developed by Russel Winder (russel@winder.org.uk)
2012-05-09 onwards

Compiler variables:
    DC - The name of the D compiler to use.  Defaults to ldc2.
    DPATH - List of paths to search for import modules.
    DVERSIONS - List of version tags to enable when compiling.
    DDEBUG - List of debug tags to enable when compiling.

Linker related variables:
    LIBS - List of library files to link in.
    DLINK - Name of the linker to use.  Defaults to ldc2.
    DLINKFLAGS - List of linker flags.

Lib tool variables:
    DLIB - Name of the lib tool to use.  Defaults to lib.
    DLIBFLAGS - List of flags to pass to the lib tool.
    LIBS - Same as for the linker. (libraries to pull into the .lib)
"""
import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Scanner.D
import SCons.Tool
import SCons.Tool.DCommon as DCommon

def generate(env):
    if False:
        print('Hello World!')
    (static_obj, shared_obj) = SCons.Tool.createObjBuilders(env)
    static_obj.add_action('.d', SCons.Defaults.DAction)
    shared_obj.add_action('.d', SCons.Defaults.ShDAction)
    static_obj.add_emitter('.d', SCons.Defaults.StaticObjectEmitter)
    shared_obj.add_emitter('.d', SCons.Defaults.SharedObjectEmitter)
    env['DC'] = env.Detect('ldc2') or 'ldc2'
    env['DCOM'] = '$DC $_DINCFLAGS $_DVERFLAGS $_DDEBUGFLAGS $_DFLAGS -c -of=$TARGET $SOURCES'
    env['_DINCFLAGS'] = '${_concat(DINCPREFIX, DPATH, DINCSUFFIX, __env__, RDirs, TARGET, SOURCE)}'
    env['_DVERFLAGS'] = '${_concat(DVERPREFIX, DVERSIONS, DVERSUFFIX, __env__)}'
    env['_DDEBUGFLAGS'] = '${_concat(DDEBUGPREFIX, DDEBUG, DDEBUGSUFFIX, __env__)}'
    env['_DFLAGS'] = '${_concat(DFLAGPREFIX, DFLAGS, DFLAGSUFFIX, __env__)}'
    env['SHDC'] = '$DC'
    env['SHDCOM'] = '$DC $_DINCFLAGS $_DVERFLAGS $_DDEBUGFLAGS $_DFLAGS -c -relocation-model=pic -of=$TARGET $SOURCES'
    env['DPATH'] = ['#/']
    env['DFLAGS'] = []
    env['DVERSIONS'] = []
    env['DDEBUG'] = []
    if env['DC']:
        DCommon.addDPATHToEnv(env, env['DC'])
    env['DINCPREFIX'] = '-I='
    env['DINCSUFFIX'] = ''
    env['DVERPREFIX'] = '-version='
    env['DVERSUFFIX'] = ''
    env['DDEBUGPREFIX'] = '-debug='
    env['DDEBUGSUFFIX'] = ''
    env['DFLAGPREFIX'] = '-'
    env['DFLAGSUFFIX'] = ''
    env['DFILESUFFIX'] = '.d'
    env['DLINK'] = '$DC'
    env['DLINKFLAGS'] = SCons.Util.CLVar('')
    env['DLINKCOM'] = '$DLINK -of=$TARGET $DLINKFLAGS $__DRPATH $SOURCES $_DLIBDIRFLAGS $_DLIBFLAGS'
    env['SHDLINK'] = '$DC'
    env['SHDLINKFLAGS'] = SCons.Util.CLVar('$DLINKFLAGS -shared -defaultlib=phobos2-ldc')
    env['SHDLINKCOM'] = '$DLINK -of=$TARGET $SHDLINKFLAGS $__SHDLIBVERSIONFLAGS $__DRPATH $SOURCES $_DLIBDIRFLAGS $_DLIBFLAGS -L-ldruntime-ldc'
    env['DLIBLINKPREFIX'] = '' if env['PLATFORM'] == 'win32' else '-L-l'
    env['DLIBLINKSUFFIX'] = '.lib' if env['PLATFORM'] == 'win32' else ''
    env['_DLIBFLAGS'] = '${_stripixes(DLIBLINKPREFIX, LIBS, DLIBLINKSUFFIX, LIBPREFIXES, LIBSUFFIXES,  __env__)}'
    env['DLIBDIRPREFIX'] = '-L-L'
    env['DLIBDIRSUFFIX'] = ''
    env['_DLIBDIRFLAGS'] = '${_concat(DLIBDIRPREFIX, LIBPATH, DLIBDIRSUFFIX, __env__, RDirs, TARGET, SOURCE)}'
    env['DLIB'] = 'lib' if env['PLATFORM'] == 'win32' else 'ar cr'
    env['DLIBCOM'] = '$DLIB $_DLIBFLAGS {0}$TARGET $SOURCES $_DLIBFLAGS'.format('-c ' if env['PLATFORM'] == 'win32' else '')
    env['DLIBFLAGPREFIX'] = '-'
    env['DLIBFLAGSUFFIX'] = ''
    env['DRPATHPREFIX'] = '-L-Wl,-rpath,' if env['PLATFORM'] == 'darwin' else '-L-rpath='
    env['DRPATHSUFFIX'] = ''
    env['_DRPATH'] = '${_concat(DRPATHPREFIX, RPATH, DRPATHSUFFIX, __env__)}'
    env['_SHDLIBVERSIONFLAGS'] = '$SHDLIBVERSIONFLAGS -L-soname=$_SHLIBSONAME'
    env['BUILDERS']['ProgramAllAtOnce'] = SCons.Builder.Builder(action='$DC $_DINCFLAGS $_DVERFLAGS $_DDEBUGFLAGS $_DFLAGS -of=$TARGET $DLINKFLAGS $__DRPATH $SOURCES $_DLIBDIRFLAGS $_DLIBFLAGS', emitter=DCommon.allAtOnceEmitter)

def exists(env):
    if False:
        print('Hello World!')
    return env.Detect('ldc2')