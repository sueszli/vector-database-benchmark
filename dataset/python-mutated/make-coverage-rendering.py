import os
import shutil
import subprocess
import sys

def main():
    if False:
        print('Hello World!')
    os.chdir(os.path.dirname(__file__))
    shutil.rmtree('coverage', ignore_errors=True)
    print('Fetching coverage files:')
    subprocess.call(['rsync', '-az', '--delete', '%s/' % os.environ['COVERAGE_DIR'], 'coverage/'])
    print('Combining coverage files:')
    os.chdir('coverage')
    print('Detect coverage file roots:')
    paths = [os.path.abspath(os.path.join(os.curdir, '..', '..'))]
    for filename in os.listdir('.'):
        if not filename.startswith('meta.coverage'):
            continue
        values = {}
        exec(open(filename).read(), values)
        if '__builtins__' in values:
            del values['__builtins__']
        paths.append(values['NUITKA_SOURCE_DIR'])
    coverage_path = os.path.abspath('.coveragerc')
    with open(coverage_path, 'w') as coverage_rcfile:
        coverage_rcfile.write('[paths]\n')
        coverage_rcfile.write('source = \n')
        for path in paths:
            coverage_rcfile.write('   ' + path + '\n')
    subprocess.call([sys.executable, '-m', 'coverage', 'combine', '--rcfile', coverage_path])
    assert os.path.exists(coverage_path)
    subprocess.call([sys.executable, '-m', 'coverage', 'html', '--rcfile', coverage_path])
if __name__ == '__main__':
    main()