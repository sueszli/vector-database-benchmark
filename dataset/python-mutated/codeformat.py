import argparse
import glob
import itertools
import os
import re
import subprocess
PATHS = ['drivers/ninaw10/*.[ch]', 'extmod/*.[ch]', 'extmod/btstack/*.[ch]', 'extmod/nimble/*.[ch]', 'lib/mbedtls_errors/tester.c', 'shared/netutils/*.[ch]', 'shared/timeutils/*.[ch]', 'shared/runtime/*.[ch]', 'shared/tinyusb/*.[ch]', 'mpy-cross/*.[ch]', 'ports/**/*.[ch]', 'py/*.[ch]']
EXCLUSIONS = ['ports/cc3200/*/*.[ch]', 'ports/esp32/managed_components/*', 'ports/nrf/boards/*.[ch]', 'ports/nrf/device/*.[ch]', 'ports/nrf/drivers/*.[ch]', 'ports/nrf/modules/ble/*.[ch]', 'ports/nrf/modules/board/*.[ch]', 'ports/nrf/modules/machine/*.[ch]', 'ports/nrf/modules/music/*.[ch]', 'ports/nrf/modules/ubluepy/*.[ch]', 'ports/nrf/modules/os/*.[ch]', 'ports/nrf/modules/time/*.[ch]', 'ports/stm32/usbdev/**/*.[ch]', 'ports/stm32/usbhost/**/*.[ch]', 'ports/*/build*']
TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UNCRUSTIFY_CFG = os.path.join(TOP, 'tools/uncrustify.cfg')

def list_files(paths, exclusions=None, prefix=''):
    if False:
        for i in range(10):
            print('nop')
    files = set()
    for pattern in paths:
        files.update(glob.glob(os.path.join(prefix, pattern), recursive=True))
    for pattern in exclusions or []:
        files.difference_update(glob.fnmatch.filter(files, os.path.join(prefix, pattern)))
    return sorted(files)

def fixup_c(filename):
    if False:
        i = 10
        return i + 15
    with open(filename) as f:
        lines = f.readlines()
    with open(filename, 'w', newline='') as f:
        dedent_stack = []
        while lines:
            l = lines.pop(0)
            m = re.match('( +)#(if |ifdef |ifndef |elif |else|endif)', l)
            if m:
                indent = len(m.group(1))
                directive = m.group(2)
                if directive in ('if ', 'ifdef ', 'ifndef '):
                    l_next = lines[0]
                    indent_next = len(re.match('( *)', l_next).group(1))
                    if indent - 4 == indent_next and re.match(' +(} else |case )', l_next):
                        l = l[4:]
                        dedent_stack.append(indent - 4)
                    else:
                        dedent_stack.append(-1)
                else:
                    if dedent_stack[-1] >= 0:
                        indent_diff = indent - dedent_stack[-1]
                        assert indent_diff >= 0
                        l = l[indent_diff:]
                    if directive == 'endif':
                        dedent_stack.pop()
            f.write(l)
        assert not dedent_stack, filename

def main():
    if False:
        print('Hello World!')
    cmd_parser = argparse.ArgumentParser(description='Auto-format C and Python files.')
    cmd_parser.add_argument('-c', action='store_true', help='Format C code only')
    cmd_parser.add_argument('-p', action='store_true', help='Format Python code only')
    cmd_parser.add_argument('-v', action='store_true', help='Enable verbose output')
    cmd_parser.add_argument('-f', action='store_true', help='Filter files provided on the command line against the default list of files to check.')
    cmd_parser.add_argument('files', nargs='*', help='Run on specific globs')
    args = cmd_parser.parse_args()
    format_c = args.c or not args.p
    format_py = args.p or not args.c
    files = []
    if args.files:
        files = list_files(args.files)
        if args.f:
            files = set((os.path.abspath(f) for f in files))
            all_files = set(list_files(PATHS, EXCLUSIONS, TOP))
            if args.v:
                for f in files - all_files:
                    print('Not checking: {}'.format(f))
            files = list(files & all_files)
    else:
        files = list_files(PATHS, EXCLUSIONS, TOP)

    def batch(cmd, N=200):
        if False:
            return 10
        files_iter = iter(files)
        while True:
            file_args = list(itertools.islice(files_iter, N))
            if not file_args:
                break
            subprocess.check_call(cmd + file_args)
    if format_c:
        command = ['uncrustify', '-c', UNCRUSTIFY_CFG, '-lC', '--no-backup']
        if not args.v:
            command.append('-q')
        batch(command)
        for file in files:
            fixup_c(file)
    if format_py:
        command = ['ruff', 'format']
        if args.v:
            command.append('-v')
        else:
            command.append('-q')
        command.append('.')
        subprocess.check_call(command, cwd=TOP)
if __name__ == '__main__':
    main()