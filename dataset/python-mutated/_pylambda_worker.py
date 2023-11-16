from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import sys
import os
from os.path import split, abspath

def get_main_dir():
    if False:
        for i in range(10):
            print('nop')
    script_path = abspath(sys.modules[__name__].__file__)
    main_dir = split(split(script_path)[0])[0]
    return main_dir

def setup_environment(info_log_function=None, error_log_function=None):
    if False:
        print('Hello World!')

    def _write_log(s, error=False):
        if False:
            for i in range(10):
                print('nop')
        if error:
            if error_log_function is None:
                print(s)
            else:
                try:
                    error_log_function(s)
                except Exception as e:
                    print('Error setting exception: ' + repr(e))
                    print('Error: %s' % str(s))
        elif info_log_function is not None:
            try:
                info_log_function(s)
            except Exception as e:
                print('Error logging info: %s.' % repr(e))
                print('Message: %s' % str(s))
    system_path = os.environ.get('__GL_SYS_PATH__', '')
    del sys.path[:]
    sys.path.extend((p.strip() for p in system_path.split(os.pathsep) if p.strip()))
    for (i, p) in enumerate(sys.path):
        _write_log('  sys.path[%d] = %s. ' % (i, sys.path[i]))
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['MKL_DOMAIN_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    main_dir = get_main_dir()
    _write_log('Main program directory: %s.' % main_dir)
    if sys.platform == 'win32':
        import ctypes
        import ctypes.wintypes as wintypes
        lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def errcheck_bool(result, func, args):
            if False:
                print('Hello World!')
            if not result:
                last_error = ctypes.get_last_error()
                if last_error != 0:
                    raise ctypes.WinError(last_error)
                else:
                    raise OSError
            return args
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.SetDllDirectoryW.errcheck = errcheck_bool
            kernel32.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,)
            kernel32.SetDllDirectoryW(lib_path)
        except Exception as e:
            _write_log('Error setting DLL load orders: %s (things may still work).\n' % str(e), error=True)
if __name__ == '__main__':
    if len(sys.argv) == 1:
        dry_run = True
    else:
        dry_run = False
    if dry_run or os.environ.get('TURI_LAMBDA_WORKER_DEBUG_MODE') == '1':
        _write_out = sys.stderr
    else:
        _write_out = None
    _write_out_file_name = os.environ.get('TURI_LAMBDA_WORKER_LOG_FILE', '')
    _write_out_file = None

    def _write_log(s, error=False):
        if False:
            for i in range(10):
                print('nop')
        s = s + '\n'
        if error:
            try:
                sys.stderr.write(s)
                sys.stderr.flush()
            except Exception:
                pass
        elif _write_out is not None:
            try:
                _write_out.write(s)
                _write_out.flush()
            except Exception:
                pass
        if _write_out_file is not None:
            try:
                _write_out_file.write(s)
                _write_out_file.flush()
            except Exception:
                pass
    if _write_out_file_name != '':
        _write_out_file_name = abspath(_write_out_file_name)
        os.environ['TURI_LAMBDA_WORKER_LOG_FILE'] = _write_out_file_name
        _write_out_file_name = _write_out_file_name + '-init'
        _write_log('Logging initialization routines to %s.' % _write_out_file_name)
        try:
            _write_out_file = open(_write_out_file_name, 'w')
        except Exception as e:
            _write_log("Error opening '%s' for write: %s" % (_write_out_file_name, repr(e)))
            _write_out_file = None
    for s in sys.argv:
        _write_log('Lambda worker args: \n  %s' % '\n  '.join(sys.argv))
    if dry_run:
        print('PyLambda script called with no IPC information; entering diagnostic mode.')
    setup_environment(info_log_function=_write_log, error_log_function=lambda s: _write_log(s, error=True))
    from turicreate._cython.cy_pylambda_workers import run_pylambda_worker
    main_dir = get_main_dir()
    default_loglevel = 5
    dryrun_loglevel = 1
    if not dry_run:
        result = run_pylambda_worker(main_dir, sys.argv[1], default_loglevel)
    else:
        result = run_pylambda_worker(main_dir, 'debug', dryrun_loglevel)
    _write_log('Lambda process exited with code %d.' % result)
    sys.exit(0)