"""A test runner for pywin32"""
import sys
import os
import distutils.sysconfig
import win32api
this_dir = os.path.dirname(__file__)
site_packages = distutils.sysconfig.get_python_lib(plat_specific=1)
if hasattr(os, 'popen3'):

    def run_test(script, cmdline_rest=''):
        if False:
            return 10
        (dirname, scriptname) = os.path.split(script)
        cwd = os.getcwd()
        os.chdir(dirname)
        try:
            executable = win32api.GetShortPathName(sys.executable)
            cmd = '%s "%s" %s' % (sys.executable, scriptname, cmdline_rest)
            print(script)
            (stdin, stdout, stderr) = os.popen3(cmd)
            stdin.close()
            while 1:
                char = stderr.read(1)
                if not char:
                    break
                sys.stdout.write(char)
            for line in stdout.readlines():
                print(line)
            stdout.close()
            result = stderr.close()
            if result is not None:
                print('****** %s failed: %s' % (script, result))
        finally:
            os.chdir(cwd)
else:
    import subprocess

    def run_test(script, cmdline_rest=''):
        if False:
            while True:
                i = 10
        (dirname, scriptname) = os.path.split(script)
        cmd = [sys.executable, '-u', scriptname] + cmdline_rest.split()
        print(script)
        popen = subprocess.Popen(cmd, shell=True, cwd=dirname, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        data = popen.communicate()[0]
        sys.stdout.buffer.write(data)
        if popen.returncode:
            print('****** %s failed: %s' % (script, popen.returncode))

def find_and_run(possible_locations, script, cmdline_rest=''):
    if False:
        i = 10
        return i + 15
    for maybe in possible_locations:
        if os.path.isfile(os.path.join(maybe, script)):
            run_test(os.path.abspath(os.path.join(maybe, script)), cmdline_rest)
            break
    else:
        raise RuntimeError("Failed to locate the test script '%s' in one of %s" % (script, possible_locations))
if __name__ == '__main__':
    maybes = [os.path.join(this_dir, 'win32', 'test'), os.path.join(site_packages, 'win32', 'test')]
    find_and_run(maybes, 'testall.py')
    maybes = [os.path.join(this_dir, 'com', 'win32com', 'test'), os.path.join(site_packages, 'win32com', 'test')]
    find_and_run(maybes, 'testall.py', '2')
    maybes = [os.path.join(this_dir, 'adodbapi', 'tests'), os.path.join(site_packages, 'adodbapi', 'tests')]
    find_and_run(maybes, 'adodbapitest.py')
    print('** The tests have some issues on py3k - not all failures are a problem...')