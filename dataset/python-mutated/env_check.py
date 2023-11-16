import psutil
import platform
import subprocess
import os
import logging
import warnings
from .utils_env import get_bytesize, _find_path, get_nano_env_var
from bigdl.nano.utils.common import _env_variable_is_set, _find_library
from psutil import cpu_count

def get_CPU_info():
    if False:
        for i in range(10):
            print('nop')
    '\n    Capture hardware information, such as CPU model, CPU informations, memory status\n    '
    socket_num = int(subprocess.getoutput('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l'))
    model_name = subprocess.getoutput('lscpu | grep "Model name"')
    model_name = model_name.partition(':')[2]
    print('>' * 20, 'Hardware Information', '>' * 20)
    print('\x1b[1m\tCPU architecture\x1b[0m:', platform.processor())
    print('\x1b[1m\tCPU model name\x1b[0m:', model_name.lstrip())
    print('\x1b[1m\tLogical Core(s)\x1b[0m:', cpu_count())
    print('\x1b[1m\tPhysical Core(s)\x1b[0m:', cpu_count(logical=False))
    print('\x1b[1m\tPhysical Core(s) per socket\x1b[0m:', int(cpu_count(logical=False) / socket_num))
    print('\x1b[1m\tSocket(s)\x1b[0m:', socket_num)
    print('\x1b[1m\tCPU usage\x1b[0m:', str(psutil.cpu_percent()) + '%')
    print('\x1b[1m\tCPU MHz\x1b[0m:', format(psutil.cpu_freq().current, '.2f'))
    print('\x1b[1m\tCPU max MHz\x1b[0m:', format(psutil.cpu_freq().max, '.2f'))
    print('\x1b[1m\tCPU min MHz\x1b[0m:', format(psutil.cpu_freq().min, '.2f'))
    print('\x1b[1m\tTotal memory\x1b[0m:', get_bytesize(psutil.virtual_memory().total))
    print('\x1b[1m\tAvailable memory\x1b[0m:', get_bytesize(psutil.virtual_memory().available))
    disabled_logo = '\x1b[0;31m✘\x1b[0m'
    abled_logo = '\x1b[0;32m✔\x1b[0m'
    for flag in ['avx512f', 'avx512_bf16', 'avx512_vnni']:
        flag_enabled = int(subprocess.getoutput(f'lscpu | grep -c {flag} '))
        if flag_enabled:
            print('\x1b[1m\tSupport\x1b[0m', flag, ':', abled_logo)
        else:
            print('\x1b[1m\tSupport\x1b[0m', flag, ':', disabled_logo)
    print('<' * 20, 'Hardware Information', '<' * 20, '\n')

def check_nano_env(use_malloc: str='tc', use_openmp: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check whether necessary environment variables are setted properly\n    '
    env_copy = os.environ.copy()
    correct_env = get_nano_env_var()
    flag = {'LD_PRELOAD': 1, 'tcmalloc': 1, 'Intel OpenMp': 1, 'TF': 1}
    name = {'LD_PRELOAD': '', 'tcmalloc': '', 'Intel OpenMp': ': ', 'TF': ': '}
    output_list = []
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'| sed 's/.*:.//g'", shell=True).splitlines()[0].decode('utf-8')
        conda_env_name = conda_dir.split('/')[-1]
    except subprocess.CalledProcessError:
        warnings.warn('Conda is not found on your computer.')
    conda_lib_dir = conda_dir + '/lib' if conda_dir is not None else None
    openmp_lib_dir = _find_library('libiomp5.so', conda_lib_dir)
    jemalloc_lib_dir = _find_library('libjemalloc.so', conda_lib_dir)
    tc_malloc_lib_dir = _find_library('libtcmalloc.so', conda_lib_dir)
    if use_openmp:
        if openmp_lib_dir is not None:
            for var in ['OMP_NUM_THREADS', 'KMP_AFFINITY', 'KMP_BLOCKTIME']:
                if not _env_variable_is_set(var, env_copy) or env_copy[var] != correct_env[var]:
                    flag['Intel OpenMp'] = 0
                    name['Intel OpenMp'] = name['Intel OpenMp'] + var + ' '
                    output_list.append('export ' + var + '=' + correct_env[var])
        else:
            output_list.append('Intel OpenMP library (libiomp5.so) is not found.')
    if use_malloc is 'je':
        if jemalloc_lib_dir is not None:
            if not _env_variable_is_set('MALLOC_CONF', env_copy) or env_copy['MALLOC_CONF'] != correct_env['MALLOC_CONF']:
                output_list.append('export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1')
        else:
            output_list.append('jemalloc library (libjemalloc.so) is not found.')
    if use_malloc is 'tc':
        if tc_malloc_lib_dir is None:
            flag['tcmalloc'] = 0
            output_list.append('tcmalloc library (libtcmalloc.so) is not found.')
    for var in ['TF_ENABLE_ONEDNN_OPTS']:
        if not _env_variable_is_set(var, env_copy) or env_copy[var] != correct_env[var]:
            flag['TF'] = 0
            name['TF'] = name['TF'] + var + ' '
            output_list.append('export ' + var + '=' + correct_env[var])
    if not _env_variable_is_set('LD_PRELOAD', env_copy) or not _find_path(env_copy['LD_PRELOAD']):
        flag['LD_PRELOAD'] = 0
        output_list.append('export LD_PRELOAD=' + correct_env['LD_PRELOAD'])
    print('>' * 20, 'Environment Variables', '>' * 20)
    disabled_logo = '\x1b[0;31mnot enabled \x1b[0m' + '\x1b[0;31m✘\x1b[0m'
    abled_logo = '\x1b[0;32m enabled \x1b[0m' + '\x1b[0;32m✔\x1b[0m'
    for category in ['LD_PRELOAD', 'tcmalloc', 'Intel OpenMp', 'TF']:
        if flag[category] == 0:
            print(f'\x1b[1m\t{category}\x1b[0m', name[category], disabled_logo)
        else:
            print(f'\x1b[1m\t{category}\x1b[0m', abled_logo)
    if output_list != []:
        print(' ')
        print('+' * 20, 'Suggested change: ', '+' * 20)
        for info in output_list:
            print(info)
        print('+' * 60, '\n')
    print('<' * 20, 'Environment Variables', '<' * 20, '\n')