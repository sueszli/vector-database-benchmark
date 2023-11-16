import subprocess
import sys
import time
import unittest
import os
import shutil
import tempfile
from manticore.binary import Elf, CGCElf
from manticore.native.mappings import mmap, munmap
from typing import List, Set
DIRPATH: str = os.path.dirname(__file__)
PYTHON_BIN: str = sys.executable

class NativeIntegrationTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        shutil.rmtree(self.test_dir)

    def _load_visited_set(self, visited: str) -> Set[int]:
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(os.path.exists(visited))
        with open(visited, 'r') as f:
            vitems = f.read().splitlines()
        return set((int(x[2:], 16) for x in vitems))

    def _run_with_timeout(self, procargs: List[str], logfile: str, timeout: int=1200) -> None:
        if False:
            print('Hello World!')
        with open(os.path.join(os.pardir, logfile), 'w') as output:
            po = subprocess.Popen(procargs, stdout=output)
            secs_used = 0
            while po.poll() is None and secs_used < timeout:
                time.sleep(1)
                sys.stderr.write('~')
                secs_used += 1
            self.assertEqual(po.returncode, 0)
            self.assertTrue(secs_used < timeout)
            sys.stderr.write('\n')

    def test_timeout(self) -> None:
        if False:
            return 10
        filename = os.path.abspath(os.path.join(DIRPATH, 'binaries', 'arguments_linux_amd64'))
        self.assertTrue(filename.startswith(os.getcwd()))
        filename = filename[len(os.getcwd()) + 1:]
        workspace = os.path.join(self.test_dir, 'workspace')
        t = time.time()
        with open(os.path.join(os.pardir, self.test_dir, 'output.log'), 'w') as output:
            subprocess.check_call(['coverage', 'run', '-m', 'manticore', '--workspace', workspace, '--core.timeout', '1', '--core.procs', '4', filename, '+++++++++'], stdout=output)
        self.assertTrue(time.time() - t < 20)

    def test_logger_verbosity(self) -> None:
        if False:
            return 10
        '\n        Tests that default verbosity produces the expected volume of output\n        '
        filename = os.path.join(DIRPATH, 'binaries', 'basic_linux_amd64')
        workspace = os.path.join(self.test_dir, 'workspace')
        output = subprocess.check_output([PYTHON_BIN, '-m', 'manticore', '--no-color', '--workspace', workspace, filename])
        output_lines = output.splitlines()
        self.assertEqual(len(output_lines), 5)
        self.assertIn(b'Loading program', output_lines[0])
        self.assertIn(b'Generated testcase No. 0 -', output_lines[1])
        self.assertIn(b'Generated testcase No. 1 -', output_lines[2])

    def _test_arguments_assertions_aux(self, binname: str, testcases_number: int, visited: List[int], add_assertion: bool=False) -> None:
        if False:
            return 10
        filename = os.path.abspath(os.path.join(DIRPATH, 'binaries', binname))
        self.assertTrue(filename.startswith(os.getcwd()))
        filename = filename[len(os.getcwd()) + 1:]
        workspace = '%s/workspace' % self.test_dir
        cmd = [PYTHON_BIN, '-m', 'manticore', '--workspace', workspace, '--core.procs', '4', '--no-color']
        if add_assertion:
            assertions = '%s/assertions.txt' % self.test_dir
            with open(assertions, 'w') as f:
                f.write('0x0000000000401003 ZF == 1')
            cmd += ['--assertions', assertions]
        cmd += [filename, '+++++++++']
        output = subprocess.check_output(cmd).splitlines()
        self.assertIn(b'm.n.manticore:INFO: Loading program', output[0])
        self.assertIn(bytes(binname, 'utf-8'), output[0])
        for i in range(testcases_number):
            line = output[1 + i]
            expected1 = b'm.c.manticore:INFO: Generated testcase No. '
            self.assertIn(expected1, line)
        actual = self._load_visited_set(os.path.join(DIRPATH, workspace, 'visited.txt'))
        self.assertLess(set(visited), actual)
        self.assertGreater(len(actual), 2000)
        self.assertEqual(len(set(visited)), len(visited))

    def test_arguments_assertions_amd64(self) -> None:
        if False:
            i = 10
            return i + 15
        self._test_arguments_assertions_aux('arguments_linux_amd64', testcases_number=1, visited=[4197952, 4197988, 4198464, 4198535, 4198850, 4198550, 4198602, 4198608, 4198619, 4417968, 4418080, 4418096, 4418112, 4418128, 4418144, 4418160, 4418208, 4418232, 4418256, 4418272, 4418391, 4198728, 4199109, 4195032, 4195057, 4788812, 4788820], add_assertion=True)

    def test_arguments_assertions_armv7(self) -> None:
        if False:
            return 10
        self._test_arguments_assertions_aux('arguments_linux_armv7', testcases_number=19, visited=[35736, 35776, 36108, 36156, 36192, 36208, 36220, 119436, 36744, 33112, 33120, 35720, 35728])

    def test_decree(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.abspath(os.path.join(DIRPATH, 'binaries', 'cadet_decree_x86'))
        self.assertTrue(filename.startswith(os.getcwd()))
        filename = filename[len(os.getcwd()) + 1:]
        workspace = os.path.join(self.test_dir, 'workspace')
        self._run_with_timeout([PYTHON_BIN, '-m', 'manticore', '--workspace', workspace, '--core.timeout', '20', '--core.procs', '4', '--no-color', '--policy', 'uncovered', filename], os.path.join(self.test_dir, 'output.log'))
        actual = self._load_visited_set(os.path.join(DIRPATH, workspace, 'visited.txt'))
        self.assertTrue(len(actual) > 100)

    def test_basic_arm(self) -> None:
        if False:
            while True:
                i = 10
        filename = os.path.abspath(os.path.join(DIRPATH, 'binaries', 'basic_linux_armv7'))
        workspace = os.path.join(self.test_dir, 'workspace')
        cmd = [PYTHON_BIN, '-m', 'manticore', '--no-color', '--workspace', workspace, filename]
        output = subprocess.check_output(cmd).splitlines()
        self.assertEqual(len(output), 5)
        self.assertIn(b'm.n.manticore:INFO: Loading program ', output[0])
        self.assertIn(b'm.c.manticore:INFO: Generated testcase No. 0 - ', output[1])
        self.assertIn(b'm.c.manticore:INFO: Generated testcase No. 1 - ', output[2])
        with open(os.path.join(workspace, 'test_00000000.stdout')) as f:
            self.assertIn('Message', f.read())
        with open(os.path.join(workspace, 'test_00000001.stdout')) as f:
            self.assertIn('Message', f.read())

    def _test_no_crash(self, test_name: str, *args) -> None:
        if False:
            while True:
                i = 10
        "\n        Tests that the specified test binary doesn't cause Manticore to crash.\n        "
        filename = os.path.abspath(os.path.join(DIRPATH, 'binaries', test_name))
        workspace = os.path.join(self.test_dir, 'workspace')
        cmd = [PYTHON_BIN, '-m', 'manticore', '--no-color', '--workspace', workspace, filename]
        cmd.extend(args)
        subprocess.check_call(cmd)

    def test_fclose_linux_amd64(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that the `fclose` example for amd64 linux doesn't crash; see #1602 and #1604.\n        "
        self._test_no_crash('fclose_linux_amd64', '+++++++')

    def test_ioctl_bogus(self) -> None:
        if False:
            return 10
        "\n        Tests that the `ioctl_bogus_linux_amd64` example for amd64 linux doesn't crash.\n        "
        self._test_no_crash('ioctl_bogus_linux_amd64')

    def test_ioctl_socket(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Tests that the `ioctl_socket_linux_amd64` example for amd64 linux doesn't crash.\n        "
        self._test_no_crash('ioctl_socket_linux_amd64')

    def test_brk_regression(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests for brk behavior. Source of brk_static_amd64:\n\n        #include <stdio.h>\n        #include <unistd.h>\n        #include <stdint.h>\n\n        int main(int argc, char *argv[]) {\n            uint8_t *p = sbrk(0);\n\n            int valid_at_first = (p == sbrk(16));\n            int valid_after_shift = ((p+16) == sbrk(0));\n            sbrk(-16);\n            int valid_after_reset = (p == sbrk(0));\n            sbrk(-(2<<20));\n            int valid_after_bad_brk = (p == sbrk(0));\n\n            if (valid_at_first && valid_after_shift && valid_after_reset && valid_after_bad_brk)\n                return 0;\n            else\n                return 1;\n        }\n        '
        filename = os.path.abspath(os.path.join(DIRPATH, 'binaries/brk_static_amd64'))
        workspace = f'{self.test_dir}/workspace'
        cmd = [PYTHON_BIN, '-m', 'manticore', '--no-color', '--workspace', workspace, filename]
        output = subprocess.check_output(cmd).splitlines()
        self.assertEqual(len(output), 4)
        self.assertIn(b'm.n.manticore:INFO: Loading program ', output[0])
        self.assertIn(b'm.c.manticore:INFO: Generated testcase No. 0 - ', output[1])

    def test_unaligned_mappings(self) -> None:
        if False:
            print('Hello World!')
        filename = os.path.join(os.path.dirname(__file__), 'binaries', 'basic_linux_amd64')
        with open(filename, 'rb') as f:
            for (addr, size) in [(1, 65534), (1, 4095), (1, 4096), (4095, 1), (4095, 2), (4095, 4096)]:
                munmap(mmap(f.fileno(), addr, size), size)
if __name__ == '__main__':
    unittest.main()