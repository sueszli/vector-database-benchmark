import os
import sys
from setup_reporter import try_execfile

def print_help():
    if False:
        print('Hello World!')
    print('%s [sub_directory] [Optional output xunit xml path prefix]\n\nThis test runs the scenario test located in the sub directory.\nEvery python file in the sub directory will be executed against pytest.\n\nOptionally, a setup.py file may exist inside the subdirectory in which case\nsetup.py is effectively "sourced" before any of the tests are run. This allows\nsetup.py to modify environment variables which will be picked up by the\ntests.\n\nFor instance, if the PATH variable is changed in setup.py a different python\nenvironment may be used to run pytest\n' % sys.argv[0])
common_dir = os.path.join(os.getcwd(), '..', 'common')
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = '%s:%s' % (os.environ['PYTHONPATH'], common_dir)
else:
    os.environ['PYTHONPATH'] = common_dir
sys.path.append(common_dir)

def process_directory(test_path, xml_prefix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Recursively processes tests in the directory `test_path`, with the following\n    logic:\n\n    * Run setup.py\n    * Run tests\n    * Recurse into subdirectory\n    * Run teardown.py\n\n    Note that inner-directory setup/teardown are run nested in between outer-\n    directory setup/teardown.\n    '
    for f in os.listdir(test_path):
        if f.endswith('.pyc'):
            try:
                os.remove(f)
            except:
                pass
    try_execfile(test_path, 'setup.py')
    exit_code = 0
    for sub_test in os.listdir(test_path):
        if sub_test != 'setup.py' and sub_test != 'teardown.py' and sub_test.endswith('.py') and (not sub_test.startswith('.')):
            xml_path = xml_prefix + '.{}.xml'.format(sub_test)
            sub_test_py = os.path.join(test_path, sub_test)
            cmd = 'pytest -v -s --junit-xml="{}" "{}"'.format(xml_path, sub_test_py)
            print(cmd)
            test_exit_code = os.system(cmd)
            exit_code |= test_exit_code
    for d in os.listdir(test_path):
        d = os.path.join(test_path, d)
        if os.path.isdir(d):
            sub_dir_exit_code = process_directory(d, xml_prefix)
            exit_code |= sub_dir_exit_code
    try_execfile(test_path, 'teardown.py')
    return exit_code

def main():
    if False:
        for i in range(10):
            print('nop')
    if len(sys.argv) < 2 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print_help()
        exit(0)
    test_path = sys.argv[1]
    if len(sys.argv) < 3:
        xml_prefix = 'tests'
    else:
        xml_prefix = sys.argv[2]
    exit_code = process_directory(test_path, xml_prefix)
    if exit_code != 0:
        exit(1)
if __name__ == '__main__':
    main()