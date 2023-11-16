import json
import os
import shutil
import subprocess
import time
import paddle

class Info:

    def __repr__(self):
        if False:
            print('Hello World!')
        return str(self.__dict__)

    def json(self):
        if False:
            return 10
        return json.dumps(self.__dict__)

    def dict(self):
        if False:
            print('Hello World!')
        return self.__dict__

    def str(self, keys=None):
        if False:
            print('Hello World!')
        if keys is None:
            keys = self.__dict__.keys()
        if isinstance(keys, str):
            keys = keys.split(',')
        values = [str(self.__dict__.get(k, '')) for k in keys]
        return ','.join(values)

def query_smi(query=None, query_type='gpu', index=None, dtype=None):
    if False:
        while True:
            i = 10
    '\n    query_type: gpu/compute\n    '
    if not has_nvidia_smi():
        return []
    cmd = ['nvidia-smi', '--format=csv,noheader,nounits']
    if isinstance(query, list) and query_type == 'gpu':
        cmd.extend(['--query-gpu={}'.format(','.join(query))])
    elif isinstance(query, list) and query_type.startswith('compute'):
        cmd.extend(['--query-compute-apps={}'.format(','.join(query))])
    else:
        return
    if isinstance(index, list) and len(index) > 0:
        cmd.extend(['--id={}'.format(','.join(index))])
    if not isinstance(dtype, list) or len(dtype) != len(query):
        dtype = [str] * len(query)
    output = subprocess.check_output(cmd, timeout=3)
    lines = output.decode('utf-8').split(os.linesep)
    ret = []
    for line in lines:
        if not line:
            continue
        info = Info()
        for (k, v, d) in zip(query, line.split(', '), dtype):
            setattr(info, k.replace('.', '_'), d(v))
        ret.append(info)
    return ret

def query_rocm_smi(query=None, index=None, dtype=None, mem=32150):
    if False:
        for i in range(10):
            print('nop')
    if not has_rocm_smi():
        return []
    cmd = ['rocm-smi']
    if not isinstance(dtype, list) or len(dtype) != len(query):
        dtype = [str] * len(query)
    output = subprocess.check_output(cmd, timeout=3)
    lines = output.decode('utf-8').split(os.linesep)
    ret = []
    for line in lines:
        if not line:
            continue
        if len(line.split()) != 8 or 'DCU' in line.split():
            continue
        info = Info()
        line = line.split()
        line = [line[0], line[7][:len(line[7]) - 1], mem, mem * float(line[6][:len(line[6]) - 1]) / 100, mem - mem * float(line[6][:len(line[6]) - 1]) / 100, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]
        for (k, v, d) in zip(query, line, dtype):
            setattr(info, k.replace('.', '_'), d(v))
        ret.append(info)
    return ret

def get_gpu_info(index=None):
    if False:
        while True:
            i = 10
    q = 'index,uuid,driver_version,name,gpu_serial,display_active,display_mode'.split(',')
    d = [int, str, str, str, str, str, str]
    index = index if index is None or isinstance(index, list) else str(index).split(',')
    return query_smi(q, index=index, dtype=d)

def get_gpu_util(index=None):
    if False:
        for i in range(10):
            print('nop')
    q = 'index,utilization.gpu,memory.total,memory.used,memory.free,timestamp'.split(',')
    d = [int, int, int, int, int, str]
    index = index if index is None or isinstance(index, list) else str(index).split(',')
    if paddle.device.is_compiled_with_rocm():
        return query_rocm_smi(q, index=index, dtype=d)
    return query_smi(q, index=index, dtype=d)

def get_gpu_process(index=None):
    if False:
        for i in range(10):
            print('nop')
    q = 'pid,process_name,gpu_uuid,gpu_name,used_memory'.split(',')
    d = [int, str, str, str, int]
    index = index if index is None or isinstance(index, list) else str(index).split(',')
    return query_smi(q, index=index, query_type='compute', dtype=d)

def has_nvidia_smi():
    if False:
        i = 10
        return i + 15
    return shutil.which('nvidia-smi')

def has_rocm_smi():
    if False:
        return 10
    return shutil.which('rocm-smi')
if __name__ == '__main__':
    print(get_gpu_info(0))
    print(get_gpu_util(0))
    print(get_gpu_process(0))
    u = get_gpu_util()
    for i in u:
        print(i.str())