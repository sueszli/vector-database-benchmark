"""Run pylint on tests.

This is needed because pylint can't check a folder which isn't a package:
https://bitbucket.org/logilab/pylint/issue/512/
"""
import os
import os.path
import sys
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from scripts import utils

def main():
    if False:
        print('Hello World!')
    'Main entry point.\n\n    Return:\n        The pylint exit status.\n    '
    utils.change_cwd()
    files = []
    for (dirpath, _dirnames, filenames) in os.walk('tests'):
        for fn in filenames:
            if os.path.splitext(fn)[1] == '.py':
                files.append(os.path.join(dirpath, fn))
    disabled = ['redefined-outer-name', 'unused-argument', 'too-many-arguments', 'missing-docstring', 'protected-access', 'len-as-condition', 'compare-to-empty-string', 'pointless-statement', 'use-implicit-booleaness-not-comparison', 'import-error', 'wrong-import-order', 'unnecessary-lambda-assignment']
    toxinidir = sys.argv[1]
    pythonpath = os.environ.get('PYTHONPATH', '').split(os.pathsep) + [toxinidir]
    args = ['--disable={}'.format(','.join(disabled)), '--ignored-modules=helpers,pytest,PyQt5', '--ignore-long-lines=(<?https?://)|^ *def [a-z]', '--method-rgx=[a-z_][A-Za-z0-9_]{1,100}$'] + sys.argv[2:] + files
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    ret = subprocess.run(['pylint'] + args, env=env, check=False).returncode
    return ret
if __name__ == '__main__':
    sys.exit(main())