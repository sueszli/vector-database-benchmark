"""
Copyright 2023, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import subprocess
MAJOR = 1
MINOR = 5
MICRO = 0
IS_RELEASED = False
IS_RELEASE_BRANCH = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def git_version():
    if False:
        while True:
            i = 10

    def _minimal_ext_cmd(cmd):
        if False:
            print('Hello World!')
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')[:7]
        prev_version_tag = '^v{}.{}.0'.format(MAJOR, MINOR - 2)
        out = _minimal_ext_cmd(['git', 'rev-list', 'HEAD', prev_version_tag, '--count'])
        COMMIT_COUNT = out.strip().decode('ascii')
        COMMIT_COUNT = '0' if not COMMIT_COUNT else COMMIT_COUNT
    except OSError:
        GIT_REVISION = 'Unknown'
        COMMIT_COUNT = 'Unknown'
    return (GIT_REVISION, COMMIT_COUNT)

def get_version_info():
    if False:
        i = 10
        return i + 15
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        (GIT_REVISION, COMMIT_COUNT) = git_version()
    elif os.path.exists('cvxpy/version.py'):
        import runpy
        ns = runpy.run_path('cvxpy/version.py')
        GIT_REVISION = ns['git_revision']
        COMMIT_COUNT = ns['git_revision']
    else:
        GIT_REVISION = 'Unknown'
        COMMIT_COUNT = 'Unknown'
    if not IS_RELEASED:
        FULLVERSION += '.dev0+' + COMMIT_COUNT + '.' + GIT_REVISION
    return (FULLVERSION, GIT_REVISION, COMMIT_COUNT)

def write_version_py(filename='cvxpy/version.py'):
    if False:
        i = 10
        return i + 15
    cnt = "\n# THIS FILE IS GENERATED FROM CVXPY SETUP.PY\nshort_version = '%(version)s'\nversion = '%(version)s'\nfull_version = '%(full_version)s'\ngit_revision = '%(git_revision)s'\ncommit_count = '%(commit_count)s'\nrelease = %(isrelease)s\nif not release:\n    version = full_version\n"
    (FULLVERSION, GIT_REVISION, COMMIT_COUNT) = get_version_info()
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'full_version': FULLVERSION, 'git_revision': GIT_REVISION, 'commit_count': COMMIT_COUNT, 'isrelease': str(IS_RELEASED)})
    finally:
        a.close()