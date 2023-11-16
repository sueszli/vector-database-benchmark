import argparse
import platform
import subprocess
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--flatc', help='path of the Flat C compiler relative to the root directory')
args = parser.parse_args()
script_path = Path(__file__).parent.resolve()
root_path = script_path.parent.parent.absolute()
flatc_exe = Path(('flatc' if not platform.system() == 'Windows' else 'flatc.exe') if not args.flatc else args.flatc)
if root_path in flatc_exe.parents:
    flatc_exe = flatc_exe.relative_to(root_path)
flatc_path = Path(root_path, flatc_exe)
assert flatc_path.exists(), 'Cannot find the flatc compiler ' + str(flatc_path)

def flatc(options, cwd=script_path):
    if False:
        print('Hello World!')
    cmd = [str(flatc_path)] + options
    subprocess.check_call(cmd, cwd=str(cwd))

def reflection_fbs_path():
    if False:
        return 10
    return Path(root_path).joinpath('reflection', 'reflection.fbs')

def make_absolute(filename, path=script_path):
    if False:
        print('Hello World!')
    return str(Path(path, filename).absolute())

def assert_file_exists(filename, path=script_path):
    if False:
        i = 10
        return i + 15
    file = Path(path, filename)
    assert file.exists(), 'could not find file: ' + filename
    return file

def assert_file_doesnt_exists(filename, path=script_path):
    if False:
        for i in range(10):
            print('nop')
    file = Path(path, filename)
    assert not file.exists(), "file exists but shouldn't: " + filename
    return file

def get_file_contents(filename, path=script_path):
    if False:
        print('Hello World!')
    file = Path(path, filename)
    contents = ''
    with open(file) as file:
        contents = file.read()
    return contents

def assert_file_contains(file, needles):
    if False:
        print('Hello World!')
    with open(file) as file:
        contents = file.read()
        for needle in [needles] if isinstance(needles, str) else needles:
            assert needle in contents, "coudn't find '" + needle + "' in file: " + str(file)
    return file

def assert_file_doesnt_contains(file, needles):
    if False:
        i = 10
        return i + 15
    with open(file) as file:
        contents = file.read()
        for needle in [needles] if isinstance(needles, str) else needles:
            assert needle not in contents, "Found unexpected '" + needle + "' in file: " + str(file)
    return file

def assert_file_and_contents(file, needle, doesnt_contain=None, path=script_path, unlink=True):
    if False:
        while True:
            i = 10
    assert_file_contains(assert_file_exists(file, path), needle)
    if doesnt_contain:
        assert_file_doesnt_contains(assert_file_exists(file, path), doesnt_contain)
    if unlink:
        Path(path, file).unlink()

def run_all(*modules):
    if False:
        for i in range(10):
            print('nop')
    failing = 0
    passing = 0
    for module in modules:
        methods = [func for func in dir(module) if callable(getattr(module, func)) and (not func.startswith('__'))]
        module_failing = 0
        module_passing = 0
        for method in methods:
            try:
                print('{0}.{1}'.format(module.__name__, method))
                getattr(module, method)(module)
                print(' [PASSED]')
                module_passing = module_passing + 1
            except Exception as e:
                print(' [FAILED]: ' + str(e))
                module_failing = module_failing + 1
        print('{0}: {1} of {2} passsed'.format(module.__name__, module_passing, module_passing + module_failing))
        passing = passing + module_passing
        failing = failing + module_failing
    return (passing, failing)