import os
import subprocess
import warnings
from bigdl.nano.utils.common import schedule_workers
from bigdl.nano.utils.common import _find_library

def get_bytesize(bytes):
    if False:
        for i in range(10):
            print('nop')
    '\n    Scale bytes to its proper format ( B / KB / MB / GB / TB / PB )\n    '
    factor = 1024
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes < factor:
            return str(format(bytes, '.2f')) + unit
        bytes /= factor

def _find_path(path_name: str) -> bool:
    if False:
        print('Hello World!')
    '\n    Find whether .so files exist under the paths or not.\n    This function will search the path one by one,\n    and confirm whether libiomp5.so and libtcmalloc.so exist or not.\n    If .so files can be found, return True. Otherwise, return False.\n    :param path_name: These paths to be found.\n    :return: True(.so files can be found) or False(not all files can be found)\n    '
    path_list = path_name.split(' ')
    libiomp5_flag = 0
    libtcmalloc_flag = 0
    for ipath in path_list:
        if os.path.exists(ipath):
            if ipath.endswith('libiomp5.so'):
                libiomp5_flag = 1
            elif ipath.endswith('libtcmalloc.so'):
                libtcmalloc_flag = 1
    return True if libiomp5_flag and libtcmalloc_flag else False

def get_nano_env_var(use_malloc: str='tc', use_openmp: bool=True, print_environment: bool=False):
    if False:
        i = 10
        return i + 15
    '\n    Return proper environment variables for jemalloc and openmp libraries.\n    :param use_malloc: Allocator to be chosen, either "je" for jemalloc or "tc" for tcmalloc.\n        default as tcmalloc.\n    :param use_openmp: If this is set to True, then use intel openmp library. Otherwise disable\n        openmp and related environment variables.\n    :param print_environment: If this is set to True, print all environment variables after\n        setting.\n    :return: Dict[str, str], indicates the key-value map of environment variables to be set by\n             nano.\n    '
    env_copy = os.environ.copy()
    nano_env = {}
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'| sed 's/.*:.//g'", shell=True).splitlines()[0].decode('utf-8')
    except subprocess.CalledProcessError:
        warnings.warn('Conda is not found on your computer.')
    conda_lib_dir = conda_dir + '/lib' if conda_dir is not None else None
    openmp_lib_dir = _find_library('libiomp5.so', conda_lib_dir)
    jemalloc_lib_dir = _find_library('libjemalloc.so', conda_lib_dir)
    tc_malloc_lib_dir = _find_library('libtcmalloc.so', conda_lib_dir)
    ld_preload_list = []
    if openmp_lib_dir is not None:
        ld_preload_list.append(openmp_lib_dir)
        cpu_procs = schedule_workers(1)
        num_threads = len(cpu_procs[0])
        nano_env['OMP_NUM_THREADS'] = str(num_threads)
        nano_env['KMP_AFFINITY'] = 'granularity=fine'
        nano_env['KMP_BLOCKTIME'] = '1'
    else:
        warnings.warn('Intel OpenMP library (libiomp5.so) is not found.')
    if jemalloc_lib_dir is not None:
        ld_preload_list.append(jemalloc_lib_dir)
        nano_env['MALLOC_CONF'] = 'oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1'
    else:
        warnings.warn('jemalloc library (libjemalloc.so) is nor found.')
    if tc_malloc_lib_dir is not None:
        ld_preload_list.append(tc_malloc_lib_dir)
    else:
        warnings.warn('tcmalloc library (libtcmalloc.so) is nor found.')
    if not use_openmp:
        nano_env.pop('OMP_NUM_THREADS')
        nano_env.pop('KMP_AFFINITY')
        nano_env.pop('KMP_BLOCKTIME')
        ld_preload_list = [lib for lib in ld_preload_list if 'libiomp5.so' not in lib]
    if use_malloc is not 'je':
        if 'MALLOC_CONF' in nano_env:
            nano_env.pop('MALLOC_CONF')
        ld_preload_list = [lib for lib in ld_preload_list if 'libjemalloc.so' not in lib]
    if use_malloc is not 'tc':
        ld_preload_list = [lib for lib in ld_preload_list if 'libtcmalloc.so' not in lib]
    nano_env['LD_PRELOAD'] = ' '.join(ld_preload_list)
    nano_env['TF_ENABLE_ONEDNN_OPTS'] = '1'
    if print_environment:
        print(nano_env)
    return nano_env