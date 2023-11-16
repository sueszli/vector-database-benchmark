import os
import glob
import shutil
import sys
home_path = os.path.join(os.path.dirname(__file__), '..', '..')
home_path = os.path.abspath(home_path)

def callback(func, path, exc_info):
    if False:
        while True:
            i = 10
    print(f'remove "{path}" failed.')

def rmtree(path):
    if False:
        for i in range(10):
            print('nop')
    if os.path.isdir(path):
        print(f'remove "{path}" recursive.')
        shutil.rmtree(path, onerror=callback)

def remove_tmpfile():
    if False:
        print('Hello World!')
    dist_file = home_path + '/dist'
    egg_file = glob.glob(home_path + '/**/*egg-info')
    rmtree(dist_file)
    for e in egg_file:
        rmtree(e)

def run_cmd(cmd):
    if False:
        i = 10
        return i + 15
    print('[CMD]', cmd)
    assert os.system(cmd) == 0
os.chdir(home_path)
remove_tmpfile()
run_cmd(f'{sys.executable} ./setup.py sdist')
run_cmd(f'{sys.executable} -m twine upload dist/*')
remove_tmpfile()