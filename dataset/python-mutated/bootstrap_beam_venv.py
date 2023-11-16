"""A utility for bootstrapping a BeamPython install.

This utility can be called with any version of Python, and attempts to create
a Python virtual environment with the requested version of Beam, and any
extra dependencies as required, installed.

The virtual environment will be placed in Apache Beam's cache directory, and
will be re-used if the parameters match.

If this script exits successfully, the last line will be the full path to a
suitable python executable.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pkg_resources import parse_version

def main():
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info < (3,):
        os.execlp('python3', 'python3', *sys.argv)
        return
    else:
        import urllib.request
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_version', help='Python major version.')
    parser.add_argument('--beam_version', help='Beam version.', default='latest')
    parser.add_argument('--extra_packages', help='Semi-colon delimited set of python dependencies.')
    parser.add_argument('--cache_dir', default=os.path.expanduser('~/.apache_beam/cache'))
    options = parser.parse_args()
    if options.python_version:
        py_version = options.python_version
        executable = 'python' + py_version
    else:
        py_version = '%s.%s' % sys.version_info[:2]
        executable = sys.executable
    if options.beam_version == 'latest':
        info = json.load(urllib.request.urlopen('https://pypi.org/pypi/apache_beam/json'))

        def maybe_strict_version(s):
            if False:
                i = 10
                return i + 15
            try:
                return parse_version(s)
            except:
                return parse_version('0.0')
        beam_version = max(info['releases'], key=maybe_strict_version)
        beam_package = 'apache_beam[gcp,aws,azure,dataframe]==' + beam_version
    elif os.path.exists(options.beam_version) or options.beam_version.startswith('http://') or options.beam_version.startswith('https://'):
        beam_version = os.path.basename(options.beam_version)
        beam_package = options.beam_version + '[gcp,aws,azure,dataframe]'
    else:
        beam_version = options.beam_version
        beam_package = 'apache_beam[gcp,aws,azure,dataframe]==' + beam_version
    deps = options.extra_packages.split(';') if options.extra_packages else []
    venv_dir = os.path.join(options.cache_dir, 'venvs', 'py-%s-beam-%s-%s' % (py_version, beam_version, hashlib.sha1(';'.join(sorted(deps)).encode('utf-8')).hexdigest()))
    venv_python = os.path.join(venv_dir, 'bin', 'python')
    if not os.path.exists(venv_python):
        try:
            subprocess.run([executable, '-m', 'venv', venv_dir], check=True)
            subprocess.run([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            subprocess.run([venv_python, '-m', 'pip', 'install', '--upgrade', 'setuptools'], check=True)
            subprocess.run([venv_python, '-m', 'pip', 'install', beam_package, 'pyparsing==2.4.2'], check=True)
            if deps:
                subprocess.run([venv_python, '-m', 'pip', 'install'] + deps, check=True)
            subprocess.run([venv_python, '-c', 'import apache_beam'], check=True)
        except:
            shutil.rmtree(venv_dir)
            raise
    print(venv_python)
if __name__ == '__main__':
    main()