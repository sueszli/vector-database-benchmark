import pathlib
import sys
import textwrap

def check(path):
    if False:
        i = 10
        return i + 15
    'Check a test file for common issues with pytest->pytorch conversion.'
    print(path.name)
    print('=' * len(path.name), '\n')
    src = path.read_text().split('\n')
    for (num, line) in enumerate(src):
        if is_comment(line):
            continue
        if line.startswith('def test'):
            report_violation(line, num, header='Module-level test function')
        if line.startswith('class Test') and 'TestCase' not in line:
            report_violation(line, num, header='Test class does not inherit from TestCase')
        if 'pytest.mark' in line:
            report_violation(line, num, header='pytest.mark.something')
        for part in ['pytest.xfail', 'pytest.skip', 'pytest.param']:
            if part in line:
                report_violation(line, num, header=f'stray {part}')
        if textwrap.dedent(line).startswith('@parametrize'):
            nn = num
            for nn in range(num, -1, -1):
                ln = src[nn]
                if 'class Test' in ln:
                    if len(ln) - len(ln.lstrip()) < 8:
                        break
            else:
                report_violation(line, num, 'off-class parametrize')
            if not src[nn - 1].startswith('@instantiate_parametrized_tests'):
                report_violation(line, num, f'missing instantiation of parametrized tests in {ln}?')

def is_comment(line):
    if False:
        print('Hello World!')
    return textwrap.dedent(line).startswith('#')

def report_violation(line, lineno, header):
    if False:
        for i in range(10):
            print('nop')
    print(f'>>>> line {lineno} : {header}\n {line}\n')
if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 2:
        raise ValueError('Usage : python check_tests_conform path/to/file/or/dir')
    path = pathlib.Path(argv[1])
    if path.is_dir():
        for this_path in path.glob('test*.py'):
            check(this_path)
    else:
        check(path)