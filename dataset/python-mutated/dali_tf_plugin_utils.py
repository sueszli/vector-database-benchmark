import subprocess
import os
import re
import sys
import fnmatch
from distutils.version import StrictVersion

def find(pattern, path):
    if False:
        print('Hello World!')
    result = []
    for (root, _, files) in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def get_module_path(module_name):
    if False:
        while True:
            i = 10
    module_path = ''
    for d in sys.path:
        possible_path = os.path.join(d, module_name)
        if os.path.isdir(possible_path) and len(d) != 0:
            module_path = possible_path
            break
    return module_path

def get_tf_compiler_version():
    if False:
        return 10
    tensorflow_libs = find('libtensorflow_framework*so*', get_module_path('tensorflow'))
    if not tensorflow_libs:
        tensorflow_libs = find('libtensorflow_framework*so*', get_module_path('tensorflow_core'))
        if not tensorflow_libs:
            return ''
    lib = tensorflow_libs[0]
    cmd = 'strings -a ' + lib + ' | grep "GCC: ("'
    s = str(subprocess.check_output(cmd, shell=True))
    lines = s.split('\\n')
    ret_ver = ''
    for line in lines:
        res = re.search('GCC:\\s*\\(.*\\)\\s*(\\d+.\\d+).\\d+', line)
        if res:
            ver = res.group(1)
            if not ret_ver or StrictVersion(ret_ver) < StrictVersion(ver):
                ret_ver = ver
    return ret_ver

def get_tf_version():
    if False:
        for i in range(10):
            print('nop')
    try:
        import pkg_resources
        s = pkg_resources.get_distribution('tensorflow-gpu').version
    except pkg_resources.DistributionNotFound:
        try:
            import tensorflow as tf
            s = tf.__version__
        except ModuleNotFoundError:
            return ''
    version = re.search('(\\d+.\\d+).\\d+', s).group(1)
    return version

def get_cpp_compiler():
    if False:
        i = 10
        return i + 15
    return os.environ.get('CXX') or 'g++'

def get_cpp_compiler_version():
    if False:
        i = 10
        return i + 15
    cmd = get_cpp_compiler() + ' --version | head -1 | grep "[c|g]++ ("'
    s = str(subprocess.check_output(cmd, shell=True).strip())
    version = re.search('[g|c]\\+\\+\\s*\\(.*\\)\\s*(\\d+.\\d+).\\d+', s).group(1)
    return version

def which(program):
    if False:
        while True:
            i = 10
    try:
        return subprocess.check_output('which ' + program, shell=True).strip()
    except subprocess.CalledProcessError:
        return None

def is_conda_env():
    if False:
        return 10
    return True if os.environ.get('CONDA_PREFIX') else False

def get_tf_build_flags():
    if False:
        while True:
            i = 10
    tf_cflags = ''
    tf_lflags = ''
    try:
        import tensorflow as tensorflow
        tf_cflags = ' '.join(tensorflow.sysconfig.get_compile_flags())
        tf_lflags = ' '.join(tensorflow.sysconfig.get_link_flags())
    except ModuleNotFoundError:
        tensorflow_path = get_module_path('tensorflow')
        if tensorflow_path != '':
            tf_cflags = ' '.join(['-I' + tensorflow_path + '/include', '-I' + tensorflow_path + '/include/external/nsync/public', '-D_GLIBCXX_USE_CXX11_ABI=0'])
            tf_lflags = ' '.join(['-L' + tensorflow_path, '-ltensorflow_framework'])
    if tf_cflags == '' and tf_lflags == '':
        raise ImportError('Could not find Tensorflow. Tensorflow must be installed before installing' + 'NVIDIA DALI TF plugin')
    return (tf_cflags, tf_lflags)

def get_dali_build_flags():
    if False:
        i = 10
        return i + 15
    dali_cflags = ''
    dali_lflags = ''
    try:
        import nvidia.dali.sysconfig as dali_sc
        dali_cflags = ' '.join(dali_sc.get_include_flags())
        dali_lflags = ' '.join(dali_sc.get_link_flags())
    except ModuleNotFoundError:
        dali_path = get_module_path('nvidia/dali')
        if dali_path != '':
            dali_cflags = ' '.join(['-I' + dali_path + '/include'])
            dali_lflags = ' '.join(['-L' + dali_path, '-ldali'])
    if dali_cflags == '' and dali_lflags == '':
        raise ImportError('Could not find DALI.')
    return (dali_cflags, dali_lflags)

def get_cuda_build_flags():
    if False:
        while True:
            i = 10
    cuda_cflags = ''
    cuda_lflags = ''
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        cuda_home = '/usr/local/cuda'
    cuda_cflags = ' '.join(['-I' + cuda_home + '/include'])
    cuda_lflags = ' '.join([])
    return (cuda_cflags, cuda_lflags)

def find_available_prebuilt_tf(requested_version, available_libs):
    if False:
        print('Hello World!')
    (req_ver_first, req_ver_second) = [int(v) for v in requested_version.split('.', 2)]
    selected_ver = None
    for file in available_libs:
        re_match = re.search('.*(\\d+)_(\\d+).*', file)
        if re_match is None:
            continue
        (ver_first, ver_second) = [int(v) for v in re_match.groups()]
        if ver_first == req_ver_first:
            if ver_second <= req_ver_second and (selected_ver is None or selected_ver < (ver_first, ver_second)):
                selected_ver = (ver_first, ver_second)
    return '.'.join([str(v) for v in selected_ver]) if selected_ver is not None else None