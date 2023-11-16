""" Caching of C compiler output.

"""
import ast
import os
import platform
import re
import sys
from collections import defaultdict
from nuitka.Tracing import scons_details_logger, scons_logger
from nuitka.utils.AppDirs import getCacheDir
from nuitka.utils.Download import getCachedDownload
from nuitka.utils.FileOperations import areSamePaths, getExternalUsePath, getFileContentByLine, getFileContents, getLinkTarget, makePath
from nuitka.utils.Importing import importFromInlineCopy
from nuitka.utils.Utils import hasMacOSIntelSupport, isMacOS, isWin32Windows
from .SconsProgress import updateSconsProgressBar
from .SconsUtils import getExecutablePath, getSconsReportValue, setEnvironmentVariable

def _getPythonDirCandidates(python_prefix):
    if False:
        while True:
            i = 10
    result = [python_prefix]
    for python_dir in (sys.prefix, os.environ.get('CONDA_PREFIX'), os.environ.get('CONDA')):
        if python_dir and python_dir not in result:
            result.append(python_dir)
    return result

def _getCcacheGuessedPaths(python_prefix):
    if False:
        i = 10
        return i + 15
    if isWin32Windows():
        for python_dir in _getPythonDirCandidates(python_prefix):
            yield os.path.join(python_dir, 'bin', 'ccache.exe')
            yield os.path.join(python_dir, 'scripts', 'ccache.exe')
    elif isMacOS():
        for python_dir in _getPythonDirCandidates(python_prefix):
            yield os.path.join(python_dir, 'bin', 'ccache')
        yield '/usr/local/opt/ccache'
        yield '/opt/homebrew/bin/ccache'

def _injectCcache(env, cc_path, python_prefix, assume_yes_for_downloads):
    if False:
        return 10
    ccache_binary = os.environ.get('NUITKA_CCACHE_BINARY')
    if ccache_binary is None:
        ccache_binary = getExecutablePath('ccache', env=env)
        if ccache_binary is None:
            for candidate in _getCcacheGuessedPaths(python_prefix):
                scons_details_logger.info("Checking if ccache is at '%s' guessed path." % candidate)
                if os.path.exists(candidate):
                    ccache_binary = candidate
                    scons_details_logger.info("Using ccache '%s' from guessed path." % ccache_binary)
                    break
        if ccache_binary is None:
            if isWin32Windows():
                url = 'https://github.com/ccache/ccache/releases/download/v4.6/ccache-4.6-windows-32.zip'
                ccache_binary = getCachedDownload(name='ccache', url=url, is_arch_specific=False, specificity=url.rsplit('/', 2)[1], flatten=True, binary='ccache.exe', message='Nuitka will make use of ccache to speed up repeated compilation.', reject=None, assume_yes_for_downloads=assume_yes_for_downloads)
            elif hasMacOSIntelSupport():
                if tuple((int(d) for d in platform.release().split('.'))) >= (18, 2):
                    url = 'https://nuitka.net/ccache/v4.2.1/ccache-4.2.1.zip'
                    ccache_binary = getCachedDownload(name='ccache', url=url, is_arch_specific=False, specificity=url.rsplit('/', 2)[1], flatten=True, binary='ccache', message='Nuitka will make use of ccache to speed up repeated compilation.', reject=None, assume_yes_for_downloads=assume_yes_for_downloads)
    else:
        scons_details_logger.info("Using ccache '%s' from NUITKA_CCACHE_BINARY environment variable." % ccache_binary)
    if ccache_binary is not None and os.path.exists(ccache_binary):
        assert areSamePaths(getExecutablePath(os.path.basename(env.the_compiler), env=env), cc_path)
        env['CXX'] = env['CC'] = '"%s" "%s"' % (ccache_binary, cc_path)
        env['LINK'] = '"%s"' % cc_path
        scons_details_logger.info("Found ccache '%s' to cache C compilation result." % ccache_binary)
        scons_details_logger.info("Providing real CC path '%s' via PATH extension." % cc_path)

def enableCcache(env, source_dir, python_prefix, assume_yes_for_downloads):
    if False:
        while True:
            i = 10
    ccache_logfile = os.path.abspath(os.path.join(source_dir, 'ccache-%d.txt' % os.getpid()))
    setEnvironmentVariable(env, 'CCACHE_LOGFILE', ccache_logfile)
    env['CCACHE_LOGFILE'] = ccache_logfile
    if 'CCACHE_DIR' not in os.environ:
        ccache_dir = os.path.join(getCacheDir(), 'ccache')
        makePath(ccache_dir)
        ccache_dir = getExternalUsePath(ccache_dir)
        setEnvironmentVariable(env, 'CCACHE_DIR', ccache_dir)
        env['CCACHE_DIR'] = ccache_dir
    if 'CLCACHE_MEMCACHED' in os.environ:
        scons_logger.warning("The setting of 'CLCACHE_MEMCACHED' environment is not supported with clcache.")
        setEnvironmentVariable(env, 'CLCACHE_MEMCACHED', None)
    setEnvironmentVariable(env, 'CCACHE_SLOPPINESS', 'include_file_ctime,include_file_mtime')
    cc_path = getExecutablePath(env.the_compiler, env=env)
    (cc_is_link, cc_link_path) = getLinkTarget(cc_path)
    if cc_is_link and os.path.basename(cc_link_path) == 'ccache':
        scons_details_logger.info('Chosen compiler %s is pointing to ccache %s already.' % (cc_path, cc_link_path))
        return True
    return _injectCcache(env=env, cc_path=cc_path, python_prefix=python_prefix, assume_yes_for_downloads=assume_yes_for_downloads)

def enableClcache(env, source_dir):
    if False:
        return 10
    if sys.version_info < (3, 5):
        return
    importFromInlineCopy('atomicwrites', must_exist=True)
    importFromInlineCopy('clcache', must_exist=True)
    import concurrent.futures.thread
    cl_binary = getExecutablePath(env.the_compiler, env)
    setEnvironmentVariable(env, 'CLCACHE_CL', cl_binary)
    env['CXX'] = env['CC'] = '<clcache>'
    setEnvironmentVariable(env, 'CLCACHE_HIDE_OUTPUTS', '1')
    if 'CLCACHE_NODIRECT' not in os.environ:
        setEnvironmentVariable(env, 'CLCACHE_NODIRECT', '1')
    clcache_stats_filename = os.path.abspath(os.path.join(source_dir, 'clcache-stats.%d.txt' % os.getpid()))
    setEnvironmentVariable(env, 'CLCACHE_STATS', clcache_stats_filename)
    env['CLCACHE_STATS'] = clcache_stats_filename
    if 'CLCACHE_DIR' not in os.environ:
        clcache_dir = os.path.join(getCacheDir(), 'clcache')
        makePath(clcache_dir)
        clcache_dir = getExternalUsePath(clcache_dir)
        setEnvironmentVariable(env, 'CLCACHE_DIR', clcache_dir)
        env['CLCACHE_DIR'] = clcache_dir
    scons_details_logger.info("Using inline copy of clcache with '%s' cl binary." % cl_binary)
    import atexit
    atexit.register(_writeClcacheStatistics)

def _writeClcacheStatistics():
    if False:
        while True:
            i = 10
    try:
        from clcache.caching import stats
        if stats is not None:
            stats.save()
    except IOError:
        raise
    except Exception:
        pass

def _getCcacheStatistics(ccache_logfile):
    if False:
        return 10
    data = {}
    if os.path.exists(ccache_logfile):
        re_command = re.compile('\\[.*? (\\d+) *\\] Command line: (.*)$')
        re_result = re.compile('\\[.*? (\\d+) *\\] Result: (.*)$')
        re_anything = re.compile('\\[.*? (\\d+) *\\] (.*)$')
        commands = {}
        for line in getFileContentByLine(ccache_logfile):
            match = re_command.match(line)
            if match:
                (pid, command) = match.groups()
                commands[pid] = command
            match = re_result.match(line)
            if match:
                (pid, result) = match.groups()
                result = result.strip()
                try:
                    command = data[commands[pid]]
                except KeyError:
                    command = 'unknown command leading to ' + line
                if result == 'unsupported compiler option':
                    if ' -o ' in command or 'unknown command' in command:
                        result = 'called for link'
                if result == 'unsupported compiler option':
                    scons_logger.warning("Encountered unsupported compiler option for ccache in '%s'." % command)
                    all_text = []
                    for line2 in getFileContentByLine(ccache_logfile):
                        match = re_anything.match(line2)
                        if match:
                            (pid2, result) = match.groups()
                            if pid == pid2:
                                all_text.append(result)
                    scons_logger.warning('Full scons output: %s' % all_text)
                if result != 'called for link':
                    data[command] = result
    return data

def checkCachingSuccess(source_dir):
    if False:
        while True:
            i = 10
    ccache_logfile = getSconsReportValue(source_dir=source_dir, key='CCACHE_LOGFILE')
    if ccache_logfile is not None:
        stats = _getCcacheStatistics(ccache_logfile)
        if not stats:
            scons_logger.warning('You are not using ccache, re-compilation of identical code will be slower than necessary. Use your OS package manager to install it.')
        else:
            counts = defaultdict(int)
            for (_command, result) in stats.items():
                if result in ('cache hit (direct)', 'cache hit (preprocessed)', 'local_storage_hit', 'primary_storage_hit'):
                    result = 'cache hit'
                elif result == 'cache_miss':
                    result = 'cache miss'
                if result in ('direct_cache_hit', 'direct_cache_miss', 'preprocessed_cache_hit', 'preprocessed_cache_miss', 'primary_storage_miss', 'called_for_link', 'local_storage_read_hit', 'local_storage_read_miss', 'local_storage_write', 'local_storage_miss', 'unsupported code directive', 'disabled'):
                    continue
                counts[result] += 1
            scons_logger.info('Compiled %d C files using ccache.' % sum(counts.values()))
            for (result, count) in sorted(counts.items()):
                scons_logger.info("Cached C files (using ccache) with result '%s': %d" % (result, count))
    if os.name == 'nt':
        clcache_stats_filename = getSconsReportValue(source_dir=source_dir, key='CLCACHE_STATS')
        if clcache_stats_filename is not None and os.path.exists(clcache_stats_filename):
            stats = ast.literal_eval(getFileContents(clcache_stats_filename))
            clcache_hit = stats['CacheHits']
            clcache_miss = stats['CacheMisses']
            scons_logger.info('Compiled %d C files using clcache with %d cache hits and %d cache misses.' % (clcache_hit + clcache_miss, clcache_hit, clcache_miss))

def runClCache(args, env):
    if False:
        i = 10
        return i + 15
    from clcache.caching import runClCache
    if str is bytes:
        scons_logger.sysexit('Error, cannot use Python2 for scons when using MSVC.')
    result = runClCache(os.environ['CLCACHE_CL'], [arg.strip('"') for arg in args[1:]], env)
    updateSconsProgressBar()
    return result