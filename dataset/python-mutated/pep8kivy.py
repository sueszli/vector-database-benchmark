import sys
from os import walk
from os.path import isdir, join, normpath
import pep8
pep8_ignores = ('E125', 'E126', 'E127', 'E128', 'E402', 'E741', 'E731', 'W503')

class KivyStyleChecker(pep8.Checker):

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        pep8.Checker.__init__(self, filename, ignore=pep8_ignores)

    def report_error(self, line_number, offset, text, check):
        if False:
            while True:
                i = 10
        return pep8.Checker.report_error(self, line_number, offset, text, check)
if __name__ == '__main__':
    print('DEPRECATED: Use pre-commit.com framework instead: ', 'pip install pre-commit && make hook')

    def usage():
        if False:
            i = 10
            return i + 15
        print('Usage: python pep8kivy.py <file_or_folder_to_check>*')
        print('Folders will be checked recursively.')
        sys.exit(1)
    if len(sys.argv) < 2:
        usage()
    elif sys.argv == 2:
        targets = sys.argv[-1]
    else:
        targets = sys.argv[-1].split()

    def check(fn):
        if False:
            while True:
                i = 10
        try:
            checker = KivyStyleChecker(fn)
        except IOError:
            return 0
        return checker.check_all()
    errors = 0
    exclude_dirs = ['kivy/lib', 'kivy/deps', 'kivy/tools/pep8checker', 'coverage', 'doc']
    exclude_dirs = [normpath(i) for i in exclude_dirs]
    exclude_files = ['kivy/gesture.py', 'kivy/tools/stub-gl-debug.py', 'kivy/modules/webdebugger.py', 'kivy/modules/_webdebugger.py']
    exclude_files = [normpath(i) for i in exclude_files]
    for target in targets:
        if isdir(target):
            for (dirpath, dirnames, filenames) in walk(target):
                cont = False
                dpath = normpath(dirpath)
                for pat in exclude_dirs:
                    if dpath.startswith(pat):
                        cont = True
                        break
                if cont:
                    continue
                for filename in filenames:
                    if not filename.endswith('.py'):
                        continue
                    cont = False
                    complete_filename = join(dirpath, filename)
                    for pat in exclude_files:
                        if complete_filename.endswith(pat):
                            cont = True
                    if cont:
                        continue
                    errors += check(complete_filename)
        else:
            for pat in exclude_dirs + exclude_files:
                if pat in target:
                    break
            else:
                if target.endswith('.py'):
                    errors += check(target)
    if errors:
        print('Error: {} style guide violation(s) encountered.'.format(errors))
        sys.exit(1)