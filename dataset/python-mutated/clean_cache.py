import os, sys, shutil
import glob
import jittor_utils as jit_utils
cache_path = os.path.join(jit_utils.home(), '.cache', 'jittor')

def callback(func, path, exc_info):
    if False:
        return 10
    print(f'remove "{path}" failed.')

def rmtree(path):
    if False:
        while True:
            i = 10
    if os.path.isdir(path):
        print(f'remove "{path}" recursive.')
        shutil.rmtree(path, onerror=callback)

def clean_all():
    if False:
        print('Hello World!')
    rmtree(cache_path)

def clean_core():
    if False:
        while True:
            i = 10
    rmtree(cache_path + '/default')
    rmtree(cache_path + '/master')
    fs = glob.glob(cache_path + '/jt*')
    for f in fs:
        rmtree(f)

def clean_cuda():
    if False:
        for i in range(10):
            print('nop')
    rmtree(cache_path + '/jtcuda')
    rmtree(cache_path + '/cutt')
    rmtree(cache_path + '/cub')
    rmtree(cache_path + '/nccl')

def clean_dataset():
    if False:
        print('Hello World!')
    rmtree(cache_path + '/dataset')

def clean_swap():
    if False:
        for i in range(10):
            print('nop')
    rmtree(cache_path + '/tmp')

def print_help():
    if False:
        i = 10
        return i + 15
    msg = '|'.join(keys)
    print(f'Usage: {sys.executable} -m jittor_utils.clean_cache [{msg}]')
    exit()
keys = [k[6:] for k in globals() if k.startswith('clean_')]
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_help()
    else:
        for k in sys.argv[1:]:
            if k not in keys:
                print_help()
            func = globals()['clean_' + k]
            func()