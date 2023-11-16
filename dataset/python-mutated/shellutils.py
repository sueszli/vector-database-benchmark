import os
import shutil
import subprocess
import sys
subprocess_check_output = subprocess.check_output

def exit_from_command_with_retcode(cmd, retcode):
    if False:
        return 10
    if retcode < 0:
        print('[error] running', ' '.join(cmd), '; process was terminated by signal', -retcode)
    else:
        print('[error] running', ' '.join(cmd), '; received return code', retcode)
    sys.exit(int(os.environ.get('CURRENT_BLOCK', 255)))

def rm_r(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given an arbitrary path, properly remove it with the correct Python construct if it exists.\n    From: http://stackoverflow.com/a/9559881\n    '
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

def run_cmd(cmd, return_output=False):
    if False:
        return 10
    '\n    Given a command as a list of arguments will attempt to execute the command\n    and, on failure, print an error message and exit.\n    '
    if not isinstance(cmd, list):
        cmd = cmd.split()
    try:
        if return_output:
            return subprocess_check_output(cmd).decode('utf-8')
        else:
            return subprocess.run(cmd, universal_newlines=True, check=True)
    except subprocess.CalledProcessError as e:
        exit_from_command_with_retcode(e.cmd, e.returncode)

def is_exe(path):
    if False:
        print('Hello World!')
    '\n    Check if a given path is an executable file.\n    From: http://stackoverflow.com/a/377028\n    '
    return os.path.isfile(path) and os.access(path, os.X_OK)

def which(program):
    if False:
        while True:
            i = 10
    "\n    Find and return the given program by its absolute path or 'None' if the program cannot be found.\n    From: http://stackoverflow.com/a/377028\n    "
    fpath = os.path.split(program)[0]
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ.get('PATH').split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None