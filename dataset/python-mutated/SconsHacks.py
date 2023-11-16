""" Hacks for scons that we apply.

We block some tools from the standard scan, there is e.g. no need to ask
what fortran version we have installed to compile with Nuitka.

Also we hack the gcc version detection to fix some bugs in it, and to avoid
scanning for g++ when we have a gcc installer, but only if that is not too
version.

"""
import os
import re
import SCons.Tool.gcc
from SCons.Script import Environment
from nuitka.Tracing import scons_details_logger
from nuitka.utils.Execution import executeProcess
from nuitka.utils.FileOperations import openTextFile
from nuitka.utils.Utils import isLinux, isMacOS
from .SconsUtils import decodeData, getExecutablePath, isGccName
v_cache = {}
_blocked_tools = ('c++', 'f95', 'f90', 'f77', 'gfortran', 'ifort', 'javah', 'tar', 'dmd', 'gdc', 'flex', 'bison', 'ranlib', 'ar', 'ldc2', 'pdflatex', 'pdftex', 'latex', 'tex', 'dvipdf', 'dvips', 'gs', 'swig', 'ifl', 'rpcgen', 'rpmbuild', 'bk', 'p4', 'm4', 'ml', 'icc', 'sccs', 'rcs', 'cvs', 'as', 'gas', 'nasm')

def _myDetectVersion(cc):
    if False:
        for i in range(10):
            print('nop')
    if isGccName(cc) or 'clang' in cc:
        command = (cc, '-dumpversion', '-dumpfullversion')
    else:
        command = (cc, '--version')
    (stdout, stderr, exit_code) = executeProcess(command)
    if exit_code != 0:
        scons_details_logger.info("Error, error exit from '%s' (%d) gave %r." % (command, exit_code, stderr))
        return None
    line = stdout.splitlines()[0]
    if str is not bytes and type(line) is bytes:
        line = decodeData(line)
    line = line.strip()
    match = re.findall('[0-9]+(?:\\.[0-9]+)+', line)
    if match:
        version = match[0]
    else:
        version = line.strip()
    version = tuple((int(part) for part in version.split('.')))
    return version

def myDetectVersion(env, cc):
    if False:
        return 10
    'Return the version of the GNU compiler, or None if it is not a GNU compiler.'
    cc = env.subst(cc)
    if not cc:
        return None
    if '++' in os.path.basename(cc):
        return None
    cc = getExecutablePath(cc, env)
    if cc is None:
        return None
    if cc not in v_cache:
        v_cache[cc] = _myDetectVersion(cc)
        scons_details_logger.info("CC '%s' version check gives %r" % (cc, v_cache[cc]))
    return v_cache[cc]

def myDetect(self, progs):
    if False:
        print('Hello World!')
    for blocked_tool in _blocked_tools:
        if blocked_tool in progs:
            return None
    return orig_detect(self, progs)
orig_detect = Environment.Detect

def getEnhancedToolDetect():
    if False:
        for i in range(10):
            print('nop')
    SCons.Tool.gcc.detect_version = myDetectVersion
    if isLinux():
        SCons.Tool.gcc.compilers.insert(0, 'x86_64-conda-linux-gnu-gcc')
    if isMacOS() and 'CONDA_TOOLCHAIN_BUILD' in os.environ:
        SCons.Tool.gcc.compilers.insert(0, '%s-clang' % os.environ['CONDA_TOOLCHAIN_BUILD'])
    return myDetect

def makeGccUseLinkerFile(source_files, module_mode, env):
    if False:
        print('Hello World!')
    tmp_linker_filename = os.path.join(env.source_dir, '@link_input.txt')
    if type(env['SHLINKCOM']) is str:
        env['SHLINKCOM'] = env['SHLINKCOM'].replace('$SOURCES', '@%s' % env.get('ESCAPE', lambda x: x)(tmp_linker_filename))
    env['LINKCOM'] = env['LINKCOM'].replace('$SOURCES', '@%s' % env.get('ESCAPE', lambda x: x)(tmp_linker_filename))
    with openTextFile(tmp_linker_filename, 'w') as tmpfile:
        for filename in source_files:
            filename = '.'.join(filename.split('.')[:-1]) + ('.os' if module_mode and os.name != 'nt' else '.o')
            if os.name == 'nt':
                filename = filename.replace(os.path.sep, '/')
            tmpfile.write('"%s"\n' % filename)
        tmpfile.write(env.subst('$SOURCES'))