"""
Functions related to core conda functionality that relates to pip

NOTE: This modules used to in conda, as conda/pip.py
"""
import json
import os
import re
import sys
from logging import getLogger
from conda.base.context import context
from conda.deprecations import deprecated
from conda.exceptions import CondaEnvException
from conda.exports import on_win
from conda.gateways.subprocess import any_subprocess
log = getLogger(__name__)

def pip_subprocess(args, prefix, cwd):
    if False:
        for i in range(10):
            print('nop')
    if on_win:
        python_path = os.path.join(prefix, 'python.exe')
    else:
        python_path = os.path.join(prefix, 'bin', 'python')
    run_args = [python_path, '-m', 'pip'] + args
    (stdout, stderr, rc) = any_subprocess(run_args, prefix, cwd=cwd)
    if not context.quiet and (not context.json):
        print('Ran pip subprocess with arguments:')
        print(run_args)
        print('Pip subprocess output:')
        print(stdout)
    if rc != 0:
        print('Pip subprocess error:', file=sys.stderr)
        print(stderr, file=sys.stderr)
        raise CondaEnvException('Pip failed')
    return (stdout, stderr)

def get_pip_installed_packages(stdout):
    if False:
        while True:
            i = 10
    'Return the list of pip packages installed based on the command output'
    m = re.search('Successfully installed\\ (.*)', stdout)
    if m:
        return m.group(1).strip().split()
    else:
        return None

@deprecated('23.9', '24.3')
def get_pip_version(prefix):
    if False:
        print('Hello World!')
    (stdout, stderr) = pip_subprocess(['-V'], prefix)
    pip_version = re.search('pip\\ (\\d+\\.\\d+\\.\\d+)', stdout)
    if not pip_version:
        raise CondaEnvException('Failed to find pip version string in output')
    else:
        pip_version = pip_version.group(1)
    return pip_version

@deprecated('23.9', '24.3')
class PipPackage(dict):

    def __str__(self):
        if False:
            while True:
                i = 10
        if 'path' in self:
            return '{} ({})-{}-<pip>'.format(self['name'], self['path'], self['version'])
        return '{}-{}-<pip>'.format(self['name'], self['version'])

@deprecated('23.9', '24.3')
def installed(prefix, output=True):
    if False:
        for i in range(10):
            print('nop')
    pip_version = get_pip_version(prefix)
    pip_major_version = int(pip_version.split('.', 1)[0])
    env = os.environ.copy()
    args = ['list']
    if pip_major_version >= 9:
        args += ['--format', 'json']
    else:
        env['PIP_FORMAT'] = 'legacy'
    try:
        (pip_stdout, stderr) = pip_subprocess(args, prefix=prefix, env=env)
    except Exception:
        if output:
            print('# Warning: subprocess call to pip failed', file=sys.stderr)
        return
    if pip_major_version >= 9:
        pkgs = json.loads(pip_stdout)
        for kwargs in pkgs:
            kwargs['name'] = kwargs['name'].lower()
            if ', ' in kwargs['version']:
                (version, path) = kwargs['version'].split(', ', 1)
                version = version.replace('-', ' ')
                kwargs['version'] = version
                kwargs['path'] = path
            yield PipPackage(**kwargs)
    else:
        pat = re.compile('([\\w.-]+)\\s+\\((.+)\\)')
        for line in pip_stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if m is None:
                if output:
                    print('Could not extract name and version from: %r' % line, file=sys.stderr)
                continue
            (name, version) = m.groups()
            name = name.lower()
            kwargs = {'name': name, 'version': version}
            if ', ' in version:
                (version, path) = version.split(', ')
                version = version.replace('-', ' ')
                kwargs.update({'path': path, 'version': version})
            yield PipPackage(**kwargs)
_canonicalize_regex = re.compile('[-_.]+')

@deprecated('23.9', '24.3')
def _canonicalize_name(name):
    if False:
        return 10
    return _canonicalize_regex.sub('-', name).lower()

@deprecated('23.9', '24.3')
def add_pip_installed(prefix, installed_pkgs, json=None, output=True):
    if False:
        while True:
            i = 10
    if isinstance(json, bool):
        output = not json
    conda_names = {_canonicalize_name(rec.name) for rec in installed_pkgs}
    for pip_pkg in installed(prefix, output=output):
        pip_name = _canonicalize_name(pip_pkg['name'])
        if pip_name in conda_names and 'path' not in pip_pkg:
            continue
        installed_pkgs.add(str(pip_pkg))