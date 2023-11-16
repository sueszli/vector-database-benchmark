import os
import subprocess
import sys
import argparse
sys.path.append('../tools')
import pyboard
CPYTHON3 = os.getenv('MICROPY_CPYTHON3', 'python3')
MICROPYTHON = os.getenv('MICROPY_MICROPYTHON', '../ports/unix/build-coverage/micropython')
NATMOD_EXAMPLE_DIR = '../examples/natmod/'
TEST_MAPPINGS = {'btree': 'btree/btree_$(ARCH).mpy', 'deflate': 'deflate/deflate_$(ARCH).mpy', 'framebuf': 'framebuf/framebuf_$(ARCH).mpy', 'heapq': 'heapq/heapq_$(ARCH).mpy', 'random': 'random/random_$(ARCH).mpy', 're': 're/re_$(ARCH).mpy'}
injected_import_hook_code = "import sys, os, io\nclass __File(io.IOBase):\n  def __init__(self):\n    self.off = 0\n  def ioctl(self, request, arg):\n    return 0\n  def readinto(self, buf):\n    buf[:] = memoryview(__buf)[self.off:self.off + len(buf)]\n    self.off += len(buf)\n    return len(buf)\nclass __FS:\n  def mount(self, readonly, mkfs):\n    pass\n  def chdir(self, path):\n    pass\n  def stat(self, path):\n    if path == '/__injected.mpy':\n      return tuple(0 for _ in range(10))\n    else:\n      raise OSError(-2) # ENOENT\n  def open(self, path, mode):\n    return __File()\nos.mount(__FS(), '/__remote')\nsys.path.insert(0, '/__remote')\nsys.modules['{}'] = __import__('__injected')\n"

class TargetSubprocess:

    def __init__(self, cmd):
        if False:
            return 10
        self.cmd = cmd

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def run_script(self, script):
        if False:
            for i in range(10):
                print('nop')
        try:
            p = subprocess.run(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, input=script)
            return (p.stdout, None)
        except subprocess.CalledProcessError as er:
            return (b'', er)

class TargetPyboard:

    def __init__(self, pyb):
        if False:
            print('Hello World!')
        self.pyb = pyb
        self.pyb.enter_raw_repl()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyb.exit_raw_repl()
        self.pyb.close()

    def run_script(self, script):
        if False:
            print('Hello World!')
        try:
            self.pyb.enter_raw_repl()
            output = self.pyb.exec_(script)
            output = output.replace(b'\r\n', b'\n')
            return (output, None)
        except pyboard.PyboardError as er:
            return (b'', er)

def run_tests(target_truth, target, args, stats):
    if False:
        while True:
            i = 10
    for test_file in args.files:
        for (k, v) in TEST_MAPPINGS.items():
            if test_file.find(k) != -1:
                test_module = k
                test_mpy = v.replace('$(ARCH)', args.arch)
                break
        else:
            print('----  {} - no matching mpy'.format(test_file))
            continue
        with open(test_file, 'rb') as f:
            test_file_data = f.read()
        test_script = b"import sys\nsys.path.remove('')\n\n"
        try:
            with open(NATMOD_EXAMPLE_DIR + test_mpy, 'rb') as f:
                test_script += b'__buf=' + bytes(repr(f.read()), 'ascii') + b'\n'
        except OSError:
            print('----  {} - mpy file not compiled'.format(test_file))
            continue
        test_script += bytes(injected_import_hook_code.format(test_module), 'ascii')
        test_script += test_file_data
        (result_out, error) = target.run_script(test_script)
        extra = ''
        if error is None and result_out == b'SKIP\n':
            result = 'SKIP'
        elif error is not None:
            result = 'FAIL'
            extra = ' - ' + str(error)
        else:
            try:
                with open(test_file + '.exp', 'rb') as f:
                    result_exp = f.read()
                error = None
            except OSError:
                (result_exp, error) = target_truth.run_script(test_file_data)
            if error is not None:
                result = 'TRUTH FAIL'
            elif result_out != result_exp:
                result = 'FAIL'
                print(result_out)
            else:
                result = 'pass'
        stats['total'] += 1
        if result == 'pass':
            stats['pass'] += 1
        elif result == 'SKIP':
            stats['skip'] += 1
        else:
            stats['fail'] += 1
        print('{:4}  {}{}'.format(result, test_file, extra))

def main():
    if False:
        for i in range(10):
            print('nop')
    cmd_parser = argparse.ArgumentParser(description='Run dynamic-native-module tests under MicroPython')
    cmd_parser.add_argument('-p', '--pyboard', action='store_true', help='run tests via pyboard.py')
    cmd_parser.add_argument('-d', '--device', default='/dev/ttyACM0', help='the device for pyboard.py')
    cmd_parser.add_argument('-a', '--arch', default='x64', help='native architecture of the target')
    cmd_parser.add_argument('files', nargs='*', help='input test files')
    args = cmd_parser.parse_args()
    target_truth = TargetSubprocess([CPYTHON3])
    if args.pyboard:
        target = TargetPyboard(pyboard.Pyboard(args.device))
    else:
        target = TargetSubprocess([MICROPYTHON])
    stats = {'total': 0, 'pass': 0, 'fail': 0, 'skip': 0}
    run_tests(target_truth, target, args, stats)
    target.close()
    target_truth.close()
    print('{} tests performed'.format(stats['total']))
    print('{} tests passed'.format(stats['pass']))
    if stats['fail']:
        print('{} tests failed'.format(stats['fail']))
    if stats['skip']:
        print('{} tests skipped'.format(stats['skip']))
    if stats['fail']:
        sys.exit(1)
if __name__ == '__main__':
    main()