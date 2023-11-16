""" Runner for standalone program tests of Nuitka.

These tests aim at showing that one specific module works in standalone
mode, trying to find issues with that packaging.

"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__.replace('\\', os.sep))), '..', '..')))
from nuitka.tools.testing.Common import checkLoadedFileAccesses, checkTestRequirements, compareWithCPython, createSearchMode, displayFileContents, displayFolderContents, displayRuntimeTraces, reportSkip, scanDirectoryForTestCases, setup, test_logger
from nuitka.tools.testing.RuntimeTracing import doesSupportTakingRuntimeTrace, getRuntimeTraceOfLoadedFiles
from nuitka.utils.FileOperations import removeDirectory
from nuitka.utils.Timing import TimerReport
from nuitka.utils.Utils import isMacOS, isWin32Windows

def displayError(dirname, filename):
    if False:
        while True:
            i = 10
    assert dirname is None
    dist_path = filename[:-3] + '.dist'
    displayFolderContents('dist folder', dist_path)
    inclusion_log_path = filename[:-3] + '.py.inclusion.log'
    displayFileContents('inclusion log', inclusion_log_path)

def main():
    if False:
        while True:
            i = 10
    python_version = setup(suite='standalone', needs_io_encoding=True)
    search_mode = createSearchMode()
    for filename in scanDirectoryForTestCases('.'):
        active = search_mode.consider(dirname=None, filename=filename)
        if not active:
            continue
        extra_flags = ['expect_success', '--standalone', 'remove_output', 'cpython_cache', 'timing', '--nowarn-mnemonic=debian-dist-packages']
        (requirements_met, error_message) = checkTestRequirements(filename)
        if not requirements_met:
            reportSkip(error_message, '.', filename)
            continue
        if filename == 'Urllib3Using.py' and os.name == 'nt':
            reportSkip('Socket module early import not working on Windows currently', '.', filename)
            continue
        if 'Idna' in filename:
            if python_version < (3,):
                extra_flags.append('ignore_stderr')
        if filename == 'GtkUsing.py':
            if python_version < (2, 7):
                reportSkip('irrelevant Python version', '.', filename)
                continue
            extra_flags.append('ignore_warnings')
        if filename.startswith('Win'):
            if os.name != 'nt':
                reportSkip('Windows only test', '.', filename)
                continue
        if filename == 'TkInterUsing.py':
            if isMacOS():
                reportSkip('Not working macOS yet', '.', filename)
                continue
            if isWin32Windows() == 'Windows':
                reportSkip('Can hang on Windows CI.', '.', filename)
                continue
            extra_flags.append('plugin_enable:tk-inter')
        if filename == 'FlaskUsing.py':
            extra_flags.append('ignore_warnings')
        if filename == 'MetadataPackagesUsing.py':
            reportSkip('MetadataPackagesUsing is environment dependent somehow, not fully working yet', '.', filename)
            continue
        if filename == 'PandasUsing.py':
            extra_flags.append('plugin_enable:no-qt')
        if filename == 'PmwUsing.py':
            extra_flags.append('plugin_enable:pmw-freezer')
        if filename == 'OpenGLUsing.py':
            extra_flags.append('ignore_warnings')
        if filename == 'PasslibUsing.py':
            extra_flags.append('ignore_warnings')
        if filename == 'Win32ComUsing.py':
            extra_flags.append('ignore_warnings')
        if filename.startswith(('PySide2', 'PySide6', 'PyQt5', 'PyQt6')):
            if python_version < (2, 7) or (3,) <= python_version < (3, 7):
                reportSkip('irrelevant Python version', '.', filename)
                continue
            if filename != 'PySide6':
                extra_flags.append('ignore_warnings')
        test_logger.info('Consider output of standalone mode compiled program: %s' % filename)
        compareWithCPython(dirname=None, filename=filename, extra_flags=extra_flags, search_mode=search_mode, needs_2to3=False, on_error=displayError)
        found_glibc_libs = []
        for dist_filename in os.listdir(os.path.join(filename[:-3] + '.dist')):
            if os.path.basename(dist_filename).startswith(('ld-linux-x86-64.so', 'libc.so.', 'libpthread.so.', 'libm.so.', 'libdl.so.', 'libBrokenLocale.so.', 'libSegFault.so', 'libanl.so.', 'libcidn.so.', 'libcrypt.so.', 'libmemusage.so', 'libmvec.so.', 'libnsl.so.', 'libnss_compat.so.', 'libnss_db.so.', 'libnss_dns.so.', 'libnss_files.so.', 'libnss_hesiod.so.', 'libnss_nis.so.', 'libnss_nisplus.so.', 'libpcprofile.so', 'libresolv.so.', 'librt.so.', 'libthread_db-1.0.so', 'libthread_db.so.', 'libutil.so.')):
                found_glibc_libs.append(dist_filename)
        if found_glibc_libs:
            test_logger.warning('Should not ship glibc libraries with the standalone executable (found %s)' % found_glibc_libs)
            sys.exit(1)
        binary_filename = os.path.join(filename[:-3] + '.dist', filename[:-3] + ('.exe' if os.name == 'nt' else '.bin'))
        try:
            if not doesSupportTakingRuntimeTrace():
                test_logger.info('Runtime traces are not possible on this machine.')
                continue
            with TimerReport('Determining run time loaded files took %.2f', logger=test_logger):
                loaded_filenames = getRuntimeTraceOfLoadedFiles(logger=test_logger, command=[binary_filename])
            illegal_accesses = checkLoadedFileAccesses(loaded_filenames=loaded_filenames, current_dir=os.getcwd())
            if illegal_accesses:
                displayError(None, filename)
                displayRuntimeTraces(test_logger, binary_filename)
                test_logger.warning("Should not access these file(s): '%r'." % illegal_accesses)
                search_mode.onErrorDetected(1)
        finally:
            removeDirectory(filename[:-3] + '.dist', ignore_errors=True)
    search_mode.finish()
if __name__ == '__main__':
    main()