"""A command line tool for building and verifying releases
Can be used for building both 'elasticsearch' and 'elasticsearchX' dists.
Only requires 'name' in 'setup.py' and the directory to be changed.
"""
import contextlib
import os
import re
import shlex
import shutil
import sys
import tempfile
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp_dir = None

@contextlib.contextmanager
def set_tmp_dir():
    if False:
        i = 10
        return i + 15
    global tmp_dir
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)
    tmp_dir = None

def run(*argv, expect_exit_code=0):
    if False:
        return 10
    global tmp_dir
    try:
        prev_dir = os.getcwd()
        if tmp_dir is None:
            os.chdir(base_dir)
        else:
            os.chdir(tmp_dir)
        cmd = ' '.join((shlex.quote(x) for x in argv))
        print('$ ' + cmd)
        exit_code = os.system(cmd)
        if exit_code != expect_exit_code:
            print('Command exited incorrectly: should have been %d was %d' % (expect_exit_code, exit_code))
            exit(exit_code or 1)
    finally:
        os.chdir(prev_dir)

def test_dist(dist):
    if False:
        while True:
            i = 10
    with set_tmp_dir() as tmp_dir:
        dist_name = re.match('^(elasticsearch\\d*)-', os.path.basename(dist)).group(1)
        run('python', '-m', 'venv', os.path.join(tmp_dir, 'venv'))
        venv_python = os.path.join(tmp_dir, 'venv/bin/python')
        run(venv_python, '-m', 'pip', 'install', '-U', 'pip', 'mypy', 'numpy', 'pandas-stubs')
        run(venv_python, '-m', 'pip', 'install', dist)
        run(venv_python, '-c', f'from {dist_name} import Elasticsearch')
        run(venv_python, '-c', f'from {dist_name}.helpers import scan, bulk, streaming_bulk, reindex')
        run(venv_python, '-c', f'from {dist_name} import Elasticsearch, AsyncElasticsearch')
        run(venv_python, '-c', f'from {dist_name}.helpers import scan, bulk, streaming_bulk, reindex, async_scan, async_bulk, async_streaming_bulk, async_reindex')
        run(venv_python, '-m', 'pip', 'install', 'aiohttp')
        run(venv_python, '-c', f'from {dist_name} import AsyncElasticsearch')
        run(venv_python, '-c', f'from {dist_name}.helpers import async_scan, async_bulk, async_streaming_bulk, async_reindex')
        if dist_name == 'elasticsearch':
            run(venv_python, '-m', 'mypy', '--strict', '--install-types', '--non-interactive', os.path.join(base_dir, 'test_elasticsearch/test_types/async_types.py'))
        for suffix in ('', '1', '2', '5', '6', '7', '8', '9', '10'):
            distx_name = f'elasticsearch{suffix}'
            run(venv_python, '-c', f'import {distx_name}', expect_exit_code=256 if distx_name != dist_name else 0)
        if dist_name == 'elasticsearch':
            run(venv_python, '-m', 'mypy', '--strict', '--install-types', '--non-interactive', os.path.join(base_dir, 'test_elasticsearch/test_types/sync_types.py'))
        else:
            run(venv_python, '-m', 'mypy', '--strict', '--install-types', '--non-interactive', os.path.join(base_dir, 'test_elasticsearch/test_types/aliased_types.py'))
        run(venv_python, '-m', 'pip', 'uninstall', '--yes', dist_name)
        run(venv_python, '-c', f'from {dist_name} import Elasticsearch', expect_exit_code=256)

def main():
    if False:
        print('Hello World!')
    run('git', 'checkout', '--', 'setup.py', 'elasticsearch/')
    run('rm', '-rf', 'build/', 'dist/*', '*.egg-info', '.eggs')
    version_path = os.path.join(base_dir, 'elasticsearch/_version.py')
    with open(version_path) as f:
        version = re.search('^__versionstr__\\s+=\\s+[\\"\\\']([^\\"\\\']+)[\\"\\\']', f.read(), re.M).group(1)
    major_version = version.split('.')[0]
    if len(sys.argv) >= 2:
        build_version = expect_version = sys.argv[1]
        if any((x in build_version for x in ('-SNAPSHOT', '-rc', '-alpha', '-beta'))):
            if '-SNAPSHOT' in build_version:
                version = version + '+dev'
            else:
                pre_number = re.search('-(a|b|rc)(?:lpha|eta|)(\\d+)$', expect_version)
                version = version + pre_number.group(1) + pre_number.group(2)
            expect_version = re.sub('(?:-(?:SNAPSHOT|alpha\\d+|beta\\d+|rc\\d+))+$', '', expect_version)
            if expect_version.endswith('.x'):
                expect_version = expect_version[:-1]
            if not version.startswith(expect_version):
                print("Version of package (%s) didn't match the expected release version (%s)" % (version, build_version))
                exit(1)
        elif expect_version != version:
            print("Version of package (%s) didn't match the expected release version (%s)" % (version, build_version))
            exit(1)
    for suffix in ('', major_version):
        run('rm', '-rf', 'build/', '*.egg-info', '.eggs')
        shutil.move(os.path.join(base_dir, 'elasticsearch'), os.path.join(base_dir, f'elasticsearch{suffix}'))
        version_path = os.path.join(base_dir, f'elasticsearch{suffix}/_version.py')
        with open(version_path) as f:
            version_data = f.read()
        version_data = re.sub('__versionstr__ = \\"[^\\"]+\\"', f'__versionstr__ = "{version}"', version_data)
        with open(version_path, 'w') as f:
            f.truncate()
            f.write(version_data)
        setup_py_path = os.path.join(base_dir, 'setup.py')
        with open(setup_py_path) as f:
            setup_py = f.read()
        with open(setup_py_path, 'w') as f:
            f.truncate()
            assert 'package_name = "elasticsearch"' in setup_py
            f.write(setup_py.replace('package_name = "elasticsearch"', f'package_name = "elasticsearch{suffix}"'))
        run('python', '-m', 'build')
        run('git', 'checkout', '--', 'setup.py', 'elasticsearch/')
        if suffix:
            run('rm', '-rf', f'elasticsearch{suffix}/')
    dists = os.listdir(os.path.join(base_dir, 'dist'))
    assert len(dists) == 4
    for dist in dists:
        test_dist(os.path.join(base_dir, 'dist', dist))
    os.system('bash -c "chmod a+w dist/*"')
    print('\n\n===============================\n\n    * Releases are ready! *\n\n$ python -m twine upload dist/*\n\n===============================')
if __name__ == '__main__':
    main()