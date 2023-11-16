from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import sys
import os
import logging
import ctypes
import glob as _glob
import subprocess as _subprocess
from ._scripts import _pylambda_worker
if sys.version_info.major == 2:
    import ConfigParser as _ConfigParser
else:
    import configparser as _ConfigParser

def make_unity_server_env():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the environment for unity_server.\n\n    The environment is necessary to start the unity_server\n    by setting the proper environments for shared libraries,\n    hadoop classpath, and module search paths for python lambda workers.\n\n    The environment has 3 components:\n    1. CLASSPATH, contains hadoop class path\n    2. __GL_PYTHON_EXECUTABLE__, path to the python executable\n    3. __GL_PYLAMBDA_SCRIPT__, path to the lambda worker executable\n    4. __GL_SYS_PATH__: contains the python sys.path of the interpreter\n    '
    env = os.environ.copy()
    classpath = get_hadoop_class_path()
    if 'CLASSPATH' in env:
        env['CLASSPATH'] = env['CLASSPATH'] + (os.path.pathsep + classpath if classpath != '' else '')
    else:
        env['CLASSPATH'] = classpath
    env['__GL_SYS_PATH__'] = os.path.pathsep.join(sys.path + [os.getcwd()])
    env['__GL_PYTHON_EXECUTABLE__'] = os.path.abspath(sys.executable)
    env['__GL_PYLAMBDA_SCRIPT__'] = os.path.abspath(_pylambda_worker.__file__)
    if 'PYTHONEXECUTABLE' in env:
        del env['PYTHONEXECUTABLE']
    if 'TURI_FILEIO_ALTERNATIVE_SSL_CERT_FILE' not in env and 'TURI_FILEIO_ALTERNATIVE_SSL_CERT_DIR' not in env:
        try:
            import certifi
            env['TURI_FILEIO_ALTERNATIVE_SSL_CERT_FILE'] = certifi.where()
            env['TURI_FILEIO_ALTERNATIVE_SSL_CERT_DIR'] = ''
        except:
            pass
    return env

def set_windows_dll_path():
    if False:
        print('Hello World!')
    '\n    Sets the dll load path so that things are resolved correctly.\n    '
    lib_path = os.path.dirname(os.path.abspath(_pylambda_worker.__file__))
    lib_path = os.path.abspath(os.path.join(lib_path, os.pardir))

    def errcheck_bool(result, func, args):
        if False:
            while True:
                i = 10
        if not result:
            last_error = ctypes.get_last_error()
            if last_error != 0:
                raise ctypes.WinError(last_error)
            else:
                raise OSError
        return args
    import ctypes.wintypes as wintypes
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetDllDirectoryW.errcheck = errcheck_bool
        kernel32.SetDllDirectoryW.argtypes = (wintypes.LPCWSTR,)
        kernel32.SetDllDirectoryW(lib_path)
    except Exception as e:
        logging.getLogger(__name__).warning('Error setting DLL load orders: %s (things should still work).' % str(e))

def get_current_platform_dll_extension():
    if False:
        print('Hello World!')
    '\n    Return the dynamic loading library extension for the current platform\n    '
    if sys.platform == 'win32':
        return 'dll'
    elif sys.platform == 'darwin':
        return 'dylib'
    else:
        return 'so'

def test_pylambda_worker():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests the pylambda workers by spawning off a separate python\n    process in order to print out additional diagnostic information\n    in case there is an error.\n    '
    import os
    from os.path import join
    from os.path import exists
    import tempfile
    import subprocess
    import datetime
    import time
    import zipfile
    import sys
    if sys.platform == 'darwin':
        if exists('/tmp'):
            tempfile.tempdir = '/tmp'
    temp_dir = tempfile.mkdtemp()
    temp_dir_sim = join(temp_dir, 'simulated')
    os.mkdir(temp_dir_sim)
    lambda_log_file_sym = join(temp_dir_sim, 'lambda_log')
    print('\nGathering installation information.')
    dir_structure_file = join(temp_dir, 'dir_structure.log')
    dir_structure_out = open(dir_structure_file, 'w')
    dump_directory_structure(dir_structure_out)
    dir_structure_out.close()
    print('\nRunning simulation.')
    env = make_unity_server_env()
    env['TURI_LAMBDA_WORKER_DEBUG_MODE'] = '1'
    env['TURI_LAMBDA_WORKER_LOG_FILE'] = lambda_log_file_sym
    proc = subprocess.Popen([sys.executable, os.path.abspath(_pylambda_worker.__file__)], env=env)
    proc.wait()
    open(join(temp_dir, 'sys_path_1.log'), 'w').write('\n'.join(('  sys.path[%d] = %s. ' % (i, p) for (i, p) in enumerate(sys.path))))
    print('\nRunning full lambda worker process')
    trial_temp_dir = join(temp_dir, 'full_run')
    os.mkdir(trial_temp_dir)
    lambda_log_file_run = join(trial_temp_dir, 'lambda_log.log')
    run_temp_dir = join(trial_temp_dir, 'run_temp_dir')
    os.mkdir(run_temp_dir)
    run_temp_dir_copy = join(temp_dir, 'run_temp_dir_copy')
    run_info_dict = {'lambda_log': lambda_log_file_run, 'temp_dir': trial_temp_dir, 'run_temp_dir': run_temp_dir, 'preserved_temp_dir': run_temp_dir_copy, 'runtime_log': join(trial_temp_dir, 'runtime.log'), 'sys_path_log': join(trial_temp_dir, 'sys_path_2.log')}
    run_script = '\nimport os\nimport traceback\nimport shutil\nimport sys\nimport glob\nfrom os.path import join\n\ndef write_exception(e):\n    ex_str = "\\n\\nException: \\n"\n    traceback_str = traceback.format_exc()\n\n    try:\n        ex_str += repr(e)\n    except Exception as e:\n        ex_str += "Error expressing exception as string."\n\n    ex_str += ": \\n" + traceback_str\n\n    try:\n        sys.stderr.write(ex_str + "\\n")\n        sys.stderr.flush()\n    except:\n        # Pretty much nothing we can do here.\n        pass\n\n# Set the system path.\nsystem_path = os.environ.get("__GL_SYS_PATH__", "")\ndel sys.path[:]\nsys.path.extend(p.strip() for p in system_path.split(os.pathsep) if p.strip())\n\ntry:\n    open(r"%(sys_path_log)s", "w").write(\n         "\\n".join("  sys.path[%%d] = %%s. " %% (i, p)\n         for i, p in enumerate(sys.path)))\nexcept Exception as e:\n    write_exception(e)\n\n\nos.environ["TURI_LAMBDA_WORKER_DEBUG_MODE"] = "1"\nos.environ["TURI_LAMBDA_WORKER_LOG_FILE"] = r"%(lambda_log)s"\nos.environ["TURI_CACHE_FILE_LOCATIONS"] = r"%(run_temp_dir)s"\nos.environ["OMP_NUM_THREADS"] = "1"\n\nimport turicreate\n\nlog_file = open(r"%(runtime_log)s", "w")\nfor k, v in turicreate.config.get_runtime_config().items():\n    log_file.write("%%s : %%s\\n" %% (str(k), str(v)))\nlog_file.close()\n\ntry:\n    sa = turicreate.SArray(range(1000))\nexcept Exception as e:\n    write_exception(e)\n\ntry:\n    print("Sum = %%d" %% (sa.apply(lambda x: x).sum()))\nexcept Exception as e:\n    write_exception(e)\n\nnew_dirs = []\ncopy_files = []\nfor root, dirs, files in os.walk(r"%(run_temp_dir)s"):\n    new_dirs += [join(root, d) for d in dirs]\n    copy_files += [join(root, name) for name in files]\n\ndef translate_name(d):\n    return os.path.abspath(join(r"%(preserved_temp_dir)s",\n                                os.path.relpath(d, r"%(run_temp_dir)s")))\n\nfor d in new_dirs:\n    try:\n        os.makedirs(translate_name(d))\n    except Exception as e:\n        sys.stderr.write("Error with: " + d)\n        write_exception(e)\n\nfor f in copy_files:\n    try:\n        shutil.copy(f, translate_name(f))\n    except Exception as e:\n        sys.stderr.write("Error with: " + f)\n        write_exception(e)\n\nfor f in glob.glob(turicreate.config.get_client_log_location() + "*"):\n    try:\n        shutil.copy(f, join(r"%(temp_dir)s", os.path.split(f)[1]))\n    except Exception as e:\n        sys.stderr.write("Error with: " + f)\n        write_exception(e)\n\n    ' % run_info_dict
    run_script_file = join(temp_dir, 'run_script.py')
    open(run_script_file, 'w').write(run_script)
    log_file_stdout = join(trial_temp_dir, 'stdout.log')
    log_file_stderr = join(trial_temp_dir, 'stderr.log')
    env = os.environ.copy()
    env['__GL_SYS_PATH__'] = os.path.pathsep.join(sys.path)
    proc = subprocess.Popen([sys.executable, os.path.abspath(run_script_file)], stdout=open(log_file_stdout, 'w'), stderr=open(log_file_stderr, 'w'), env=env)
    proc.wait()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    zipfile_name = join(temp_dir, 'testing_logs-%d-%s.zip' % (os.getpid(), timestamp))
    print('Creating archive of log files in %s.' % zipfile_name)
    save_files = []
    for (root, dirs, files) in os.walk(temp_dir):
        save_files += [join(root, name) for name in files]
    with zipfile.ZipFile(zipfile_name, 'w') as logzip:
        error_logs = []
        for f in save_files:
            try:
                logzip.write(f)
            except Exception as e:
                error_logs.append('%s: error = %s' % (f, repr(e)))
        if error_logs:
            error_log_file = join(temp_dir, 'archive_errors.log')
            open(error_log_file, 'w').write('\n\n'.join(error_logs))
            logzip.write(error_log_file)
    print('################################################################################')
    print('#   ')
    print('#   Results of lambda test logged as %s.' % zipfile_name)
    print('#   ')
    print('################################################################################')
    print('Cleaning up.')
    for f in save_files:
        try:
            os.remove(f)
        except Exception:
            print('Could not delete: %s' % f)

def dump_directory_structure(out=sys.stdout):
    if False:
        i = 10
        return i + 15
    '\n    Dumps a detailed report of the turicreate directory structure\n    and files, along with the output of os.lstat for each.  This is useful\n    for debugging purposes.\n    '
    'Dumping Installation Directory Structure for Debugging: '
    import sys, os
    from os.path import split, abspath, join
    from itertools import chain
    main_dir = split(abspath(sys.modules[__name__].__file__))[0]
    visited_files = []

    def on_error(err):
        if False:
            for i in range(10):
                print('nop')
        visited_files.append(('  ERROR', str(err)))
    for (path, dirs, files) in os.walk(main_dir, onerror=on_error):
        for fn in chain(files, dirs):
            name = join(path, fn)
            try:
                visited_files.append((name, repr(os.lstat(name))))
            except:
                visited_files.append((name, 'ERROR calling os.lstat.'))

    def strip_name(n):
        if False:
            i = 10
            return i + 15
        if n[:len(main_dir)] == main_dir:
            return '<root>/' + n[len(main_dir):]
        else:
            return n
    out.write('\n'.join(('  %s: %s' % (strip_name(name), stats) for (name, stats) in sorted(visited_files))))
    out.flush()
__hadoop_class_warned = False

def get_hadoop_class_path():
    if False:
        for i in range(10):
            print('nop')
    env = os.environ.copy()
    hadoop_exe_name = 'hadoop'
    if sys.platform == 'win32':
        hadoop_exe_name += '.cmd'
    output = None
    try:
        try:
            output = _subprocess.check_output([hadoop_exe_name, 'classpath']).decode()
        except:
            output = _subprocess.check_output(['/'.join([env['HADOOP_HOME'], 'bin', hadoop_exe_name]), 'classpath']).decode()
        output = os.path.pathsep.join((os.path.realpath(path) for path in output.split(os.path.pathsep)))
        return _get_expanded_classpath(output)
    except Exception as e:
        global __hadoop_class_warned
        if not __hadoop_class_warned:
            __hadoop_class_warned = True
            logging.getLogger(__name__).debug('Exception trying to retrieve Hadoop classpath: %s' % e)
    logging.getLogger(__name__).debug('Hadoop not found. HDFS url is not supported. Please make hadoop available from PATH or set the environment variable HADOOP_HOME.')
    return ''

def _get_expanded_classpath(classpath):
    if False:
        while True:
            i = 10
    '\n    Take a classpath of the form:\n      /etc/hadoop/conf:/usr/lib/hadoop/lib/*:/usr/lib/hadoop/.//*: ...\n\n    and return it expanded to all the JARs (and nothing else):\n      /etc/hadoop/conf:/usr/lib/hadoop/lib/netty-3.6.2.Final.jar:/usr/lib/hadoop/lib/jaxb-api-2.2.2.jar: ...\n\n    mentioned in the path\n    '
    if classpath is None or classpath == '':
        return ''
    jars = os.path.pathsep.join((os.path.pathsep.join([os.path.abspath(jarpath) for jarpath in _glob.glob(path)]) for path in classpath.split(os.path.pathsep)))
    logging.getLogger(__name__).debug('classpath being used: %s' % jars)
    return jars

def get_config_file():
    if False:
        while True:
            i = 10
    '\n    Returns the file name of the config file from which the environment\n    variables are written.\n    '
    import os
    from os.path import abspath, expanduser, join, exists
    __lib_name = 'turicreate'
    __default_config_path = join(expanduser('~'), '.%s' % __lib_name, 'config')
    if 'TURI_CONFIG_FILE' in os.environ:
        __default_config_path = abspath(expanduser(os.environ['TURI_CONFIG_FILE']))
        if not exists(__default_config_path):
            print("WARNING: Config file specified in environment variable 'TURI_CONFIG_FILE' as '%s', but this path does not exist." % __default_config_path)
    return __default_config_path

def setup_environment_from_config_file():
    if False:
        while True:
            i = 10
    '\n    Imports the environmental configuration settings from the\n    config file, if present, and sets the environment\n    variables to test it.\n    '
    from os.path import exists
    config_file = get_config_file()
    if not exists(config_file):
        return
    try:
        config = _ConfigParser.SafeConfigParser()
        config.read(config_file)
        __section = 'Environment'
        if config.has_section(__section):
            items = config.items(__section)
            for (k, v) in items:
                try:
                    os.environ[k.upper()] = v
                except Exception as e:
                    print("WARNING: Error setting environment variable '%s = %s' from config file '%s': %s." % (k, str(v), config_file, str(e)))
    except Exception as e:
        print("WARNING: Error reading config file '%s': %s." % (config_file, str(e)))

def write_config_file_value(key, value):
    if False:
        print('Hello World!')
    '\n    Writes an environment variable configuration to the current\n    config file.  This will be read in on the next restart.\n    The config file is created if not present.\n\n    Note: The variables will not take effect until after restart.\n    '
    filename = get_config_file()
    config = _ConfigParser.SafeConfigParser()
    config.read(filename)
    __section = 'Environment'
    if not config.has_section(__section):
        config.add_section(__section)
    config.set(__section, key, value)
    with open(filename, 'w') as config_file:
        config.write(config_file)