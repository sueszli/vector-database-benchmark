import os
import re
import sys
import shutil
import warnings
import textwrap
import unittest
import tempfile
import subprocess
from distutils import ccompiler
import runtests
import Cython.Distutils.extension
import Cython.Distutils.old_build_ext as build_ext
from Cython.Debugger import Cygdb as cygdb
root = os.path.dirname(os.path.abspath(__file__))
codefile = os.path.join(root, 'codefile')
cfuncs_file = os.path.join(root, 'cfuncs.c')
with open(codefile) as f:
    source_to_lineno = dict(((line.strip(), i + 1) for (i, line) in enumerate(f)))
have_gdb = None

def test_gdb():
    if False:
        while True:
            i = 10
    global have_gdb
    if have_gdb is not None:
        return have_gdb
    have_gdb = False
    try:
        p = subprocess.Popen(['gdb', '-nx', '--version'], stdout=subprocess.PIPE)
    except OSError:
        gdb_version = None
    else:
        (stdout, _) = p.communicate()
        regex = 'GNU gdb [^\\d]*(\\d+)\\.(\\d+)'
        gdb_version = re.match(regex, stdout.decode('ascii', 'ignore'))
    if gdb_version:
        gdb_version_number = list(map(int, gdb_version.groups()))
        if gdb_version_number >= [7, 2]:
            have_gdb = True
            with tempfile.NamedTemporaryFile(mode='w+') as python_version_script:
                python_version_script.write('python import sys; print("%s %s" % sys.version_info[:2])')
                python_version_script.flush()
                p = subprocess.Popen(['gdb', '-batch', '-x', python_version_script.name], stdout=subprocess.PIPE)
                (stdout, _) = p.communicate()
                try:
                    internal_python_version = list(map(int, stdout.decode('ascii', 'ignore').split()))
                    if internal_python_version < [2, 7]:
                        have_gdb = False
                except ValueError:
                    have_gdb = False
    if not have_gdb:
        warnings.warn('Skipping gdb tests, need gdb >= 7.2 with Python >= 2.7')
    return have_gdb

class DebuggerTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Run gdb and have cygdb import the debug information from the code\n        defined in TestParseTreeTransforms's setUp method\n        "
        if not test_gdb():
            return
        self.tempdir = tempfile.mkdtemp()
        self.destfile = os.path.join(self.tempdir, 'codefile.pyx')
        self.debug_dest = os.path.join(self.tempdir, 'cython_debug', 'cython_debug_info_codefile')
        self.cfuncs_destfile = os.path.join(self.tempdir, 'cfuncs')
        self.cwd = os.getcwd()
        try:
            os.chdir(self.tempdir)
            shutil.copy(codefile, self.destfile)
            shutil.copy(cfuncs_file, self.cfuncs_destfile + '.c')
            shutil.copy(cfuncs_file.replace('.c', '.h'), self.cfuncs_destfile + '.h')
            compiler = ccompiler.new_compiler()
            compiler.compile(['cfuncs.c'], debug=True, extra_postargs=['-fPIC'])
            opts = dict(test_directory=self.tempdir, module='codefile', module_path=self.destfile)
            optimization_disabler = build_ext.Optimization()
            cython_compile_testcase = runtests.CythonCompileTestCase(workdir=self.tempdir, cleanup_workdir=False, tags=runtests.parse_tags(codefile), **opts)
            new_stderr = open(os.devnull, 'w')
            stderr = sys.stderr
            sys.stderr = new_stderr
            optimization_disabler.disable_optimization()
            try:
                cython_compile_testcase.run_cython(targetdir=self.tempdir, incdir=None, annotate=False, extra_compile_options={'gdb_debug': True, 'output_dir': self.tempdir}, **opts)
                cython_compile_testcase.run_distutils(test_directory=opts['test_directory'], module=opts['module'], workdir=opts['test_directory'], incdir=None, extra_extension_args={'extra_objects': ['cfuncs.o']})
            finally:
                optimization_disabler.restore_state()
                sys.stderr = stderr
                new_stderr.close()
        except:
            os.chdir(self.cwd)
            raise

    def tearDown(self):
        if False:
            print('Hello World!')
        if not test_gdb():
            return
        os.chdir(self.cwd)
        shutil.rmtree(self.tempdir)

class GdbDebuggerTestCase(DebuggerTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if not test_gdb():
            return
        super(GdbDebuggerTestCase, self).setUp()
        prefix_code = textwrap.dedent('            python\n\n            import os\n            import sys\n            import traceback\n\n            def excepthook(type, value, tb):\n                traceback.print_exception(type, value, tb)\n                sys.stderr.flush()\n                sys.stdout.flush()\n                os._exit(1)\n\n            sys.excepthook = excepthook\n\n            # Have tracebacks end up on sys.stderr (gdb replaces sys.stderr\n            # with an object that calls gdb.write())\n            sys.stderr = sys.__stderr__\n\n            end\n            ')
        code = textwrap.dedent('            python\n\n            from Cython.Debugger.Tests import test_libcython_in_gdb\n            test_libcython_in_gdb.main(version=%r)\n\n            end\n            ' % (sys.version_info[:2],))
        self.gdb_command_file = cygdb.make_command_file(self.tempdir, prefix_code)
        with open(self.gdb_command_file, 'a') as f:
            f.write(code)
        args = ['gdb', '-batch', '-x', self.gdb_command_file, '-n', '--args', sys.executable, '-c', 'import codefile']
        paths = []
        path = os.environ.get('PYTHONPATH')
        if path:
            paths.append(path)
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(Cython.__file__))))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
        self.p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def tearDown(self):
        if False:
            print('Hello World!')
        if not test_gdb():
            return
        try:
            super(GdbDebuggerTestCase, self).tearDown()
            if self.p:
                try:
                    self.p.stdout.close()
                except:
                    pass
                try:
                    self.p.stderr.close()
                except:
                    pass
                self.p.wait()
        finally:
            os.remove(self.gdb_command_file)

class TestAll(GdbDebuggerTestCase):

    def test_all(self):
        if False:
            print('Hello World!')
        if not test_gdb():
            return
        (out, err) = self.p.communicate()
        out = out.decode('UTF-8')
        err = err.decode('UTF-8')
        exit_status = self.p.returncode
        if exit_status == 1:
            sys.stderr.write(out)
            sys.stderr.write(err)
        elif exit_status >= 2:
            border = u'*' * 30
            start = u'%s   v INSIDE GDB v   %s' % (border, border)
            stderr = u'%s   v STDERR v   %s' % (border, border)
            end = u'%s   ^ INSIDE GDB ^   %s' % (border, border)
            errmsg = u'\n%s\n%s%s\n%s%s' % (start, out, stderr, err, end)
            sys.stderr.write(errmsg)
if __name__ == '__main__':
    unittest.main()