"""
The Cython debugger

The current directory should contain a directory named 'cython_debug', or a
path to the cython project directory should be given (the parent directory of
cython_debug).

Additional gdb args can be provided only if a path to the project directory is
given.
"""
import os
import sys
import glob
import tempfile
import textwrap
import subprocess
import optparse
import logging
logger = logging.getLogger(__name__)

def make_command_file(path_to_debug_info, prefix_code='', no_import=False, skip_interpreter=False):
    if False:
        return 10
    if not no_import:
        pattern = os.path.join(path_to_debug_info, 'cython_debug', 'cython_debug_info_*')
        debug_files = glob.glob(pattern)
        if not debug_files:
            sys.exit('%s.\nNo debug files were found in %s. Aborting.' % (usage, os.path.abspath(path_to_debug_info)))
    (fd, tempfilename) = tempfile.mkstemp()
    f = os.fdopen(fd, 'w')
    try:
        f.write(prefix_code)
        f.write(textwrap.dedent('            # This is a gdb command file\n            # See https://sourceware.org/gdb/onlinedocs/gdb/Command-Files.html\n\n            set breakpoint pending on\n            set print pretty on\n\n            python\n            try:\n                # Activate virtualenv, if we were launched from one\n                import os\n                virtualenv = os.getenv(\'VIRTUAL_ENV\')\n                if virtualenv:\n                    path_to_activate_this_py = os.path.join(virtualenv, \'bin\', \'activate_this.py\')\n                    print("gdb command file: Activating virtualenv: %s; path_to_activate_this_py: %s" % (\n                        virtualenv, path_to_activate_this_py))\n                    with open(path_to_activate_this_py) as f:\n                        exec(f.read(), dict(__file__=path_to_activate_this_py))\n                from Cython.Debugger import libcython, libpython\n            except Exception as ex:\n                from traceback import print_exc\n                print("There was an error in Python code originating from the file ' + str(__file__) + '")\n                print("It used the Python interpreter " + str(sys.executable))\n                print_exc()\n                exit(1)\n            end\n            '))
        if no_import:
            pass
        else:
            if not skip_interpreter:
                path = os.path.join(path_to_debug_info, 'cython_debug', 'interpreter')
                interpreter_file = open(path)
                try:
                    interpreter = interpreter_file.read()
                finally:
                    interpreter_file.close()
                f.write('file %s\n' % interpreter)
            f.write('\n'.join(('cy import %s\n' % fn for fn in debug_files)))
            if not skip_interpreter:
                f.write(textwrap.dedent('                    python\n                    import sys\n                    try:\n                        gdb.lookup_type(\'PyModuleObject\')\n                    except RuntimeError:\n                        sys.stderr.write(\n                            "' + interpreter + ' was not compiled with debug symbols (or it was "\n                            "stripped). Some functionality may not work (properly).\\n")\n                    end\n                '))
            f.write('source .cygdbinit')
    finally:
        f.close()
    return tempfilename
usage = 'Usage: cygdb [options] [PATH [-- GDB_ARGUMENTS]]'

def main(path_to_debug_info=None, gdb_argv=None, no_import=False):
    if False:
        print('Hello World!')
    "\n    Start the Cython debugger. This tells gdb to import the Cython and Python\n    extensions (libcython.py and libpython.py) and it enables gdb's pending\n    breakpoints.\n\n    path_to_debug_info is the path to the Cython build directory\n    gdb_argv is the list of options to gdb\n    no_import tells cygdb whether it should import debug information\n    "
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--gdb-executable', dest='gdb', default='gdb', help='gdb executable to use [default: gdb]')
    parser.add_option('--verbose', '-v', dest='verbosity', action='count', default=0, help='Verbose mode. Multiple -v options increase the verbosity')
    parser.add_option('--skip-interpreter', dest='skip_interpreter', default=False, action='store_true', help='Do not automatically point GDB to the same interpreter used to generate debugging information')
    (options, args) = parser.parse_args()
    if path_to_debug_info is None:
        if len(args) > 1:
            path_to_debug_info = args[0]
        else:
            path_to_debug_info = os.curdir
    if gdb_argv is None:
        gdb_argv = args[1:]
    if path_to_debug_info == '--':
        no_import = True
    logging_level = logging.WARN
    if options.verbosity == 1:
        logging_level = logging.INFO
    if options.verbosity >= 2:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)
    skip_interpreter = options.skip_interpreter
    logger.info('verbosity = %r', options.verbosity)
    logger.debug('options = %r; args = %r', options, args)
    logger.debug('Done parsing command-line options. path_to_debug_info = %r, gdb_argv = %r', path_to_debug_info, gdb_argv)
    tempfilename = make_command_file(path_to_debug_info, no_import=no_import, skip_interpreter=skip_interpreter)
    logger.info('Launching %s with command file: %s and gdb_argv: %s', options.gdb, tempfilename, gdb_argv)
    with open(tempfilename) as tempfile:
        logger.debug('Command file (%s) contains: """\n%s"""', tempfilename, tempfile.read())
        logger.info('Spawning %s...', options.gdb)
        p = subprocess.Popen([options.gdb, '-command', tempfilename] + gdb_argv)
        logger.info('Spawned %s (pid %d)', options.gdb, p.pid)
        while True:
            try:
                logger.debug('Waiting for gdb (pid %d) to exit...', p.pid)
                ret = p.wait()
                logger.debug('Wait for gdb (pid %d) to exit is done. Returned: %r', p.pid, ret)
            except KeyboardInterrupt:
                pass
            else:
                break
        logger.debug('Closing temp command file with fd: %s', tempfile.fileno())
    logger.debug('Removing temp command file: %s', tempfilename)
    os.remove(tempfilename)
    logger.debug('Removed temp command file: %s', tempfilename)