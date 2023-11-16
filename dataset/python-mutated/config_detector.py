"""Auto-detects machine configurations and outputs the results to shell or file.

Supports linux only currently.

Usage:
  python config_detector.py [--save_output] [--filename] [--debug]

Example command:
  python config_detector.py --save_output=True --filename=configs.json
  --debug=False

Flag option(s):
  save_output  (True | False)       Save output to a file.
                                    (Default: True)
  filename     <file_name>.json     Filename(.json) for storing configs.
                                    (Default: `configs.json`)
  debug        (True | False)       View debug and stderr messages.
                                    (Default: False)

The following machine configuration will be detected:
  Platform              Operating system (linux | macos | windows)
  CPU                   CPU type (e.g. `GenuineIntel`)
  CPU architecture      Processor type (32-bit | 64-bit)
  CPU ISA               CPU instruction set (e.g. `sse4`, `sse4_1`, `avx`)
  Distribution          Operating system distribution (e.g. Ubuntu)
  Distribution version  Operating system distribution version (e.g. 14.04)
  GPU                   GPU type (e.g. `Tesla K80`)
  GPU count             Number of GPU's available
  CUDA version          CUDA version by default (e.g. `10.1`)
  CUDA version all      CUDA version(s) all available
  cuDNN version         cuDNN version (e.g. `7.5.0`)
  GCC version           GCC version (e.g. `7.3.0`)
  GLIBC version         GLIBC version (e.g. `2.24`)
  libstdc++ version     libstdc++ version (e.g. `3.4.25`)

Output:
  Shell output (print)
      A table containing status and info on all configurations will be
      printed out to shell.

  Configuration file (.json):
      Depending on `--save_output` option, this script outputs a .json file
      (in the same directory) containing all user machine configurations
      that were detected.
"""
import collections
import json
import re
import subprocess
import sys
from absl import app
from absl import flags
from tensorflow.tools.tensorflow_builder.config_detector.data import cuda_compute_capability
FLAGS = flags.FLAGS
flags.DEFINE_boolean('save_output', True, 'Save output to a file. [True/False]')
flags.DEFINE_string('filename', 'configs.json', 'Output filename.')
flags.DEFINE_boolean('debug', False, 'View debug messages. [True/False]')
cmds_linux = {'cpu_type': "cat /proc/cpuinfo 2>&1 | grep 'vendor' | uniq", 'cpu_arch': 'uname -m', 'distrib': "cat /etc/*-release | grep DISTRIB_ID* | sed 's/^.*=//'", 'distrib_ver': "cat /etc/*-release | grep DISTRIB_RELEASE* | sed 's/^.*=//'", 'gpu_type': "sudo lshw -C display | grep product:* | sed 's/^.*: //'", 'gpu_type_no_sudo': "lspci | grep 'VGA compatible\\|3D controller' | cut -d' ' -f 1 | xargs -i lspci -v -s {} | head -n 2 | tail -1 | awk '{print $(NF-2), $(NF-1), $NF}'", 'gpu_count': 'sudo lshw -C display | grep *-display:* | wc -l', 'gpu_count_no_sudo': "lspci | grep 'VGA compatible\\|3D controller' | wc -l", 'cuda_ver_all': 'ls -d /usr/local/cuda* 2> /dev/null', 'cuda_ver_dflt': ['nvcc --version 2> /dev/null', "cat /usr/local/cuda/version.txt 2> /dev/null | awk '{print $NF}'"], 'cudnn_ver': ['whereis cudnn.h', "cat `awk '{print $2}'` | grep CUDNN_MAJOR -A 2 | echo `awk '{print $NF}'` | awk '{print $1, $2, $3}' | sed 's/ /./g'"], 'gcc_ver': "gcc --version | awk '{print $NF}' | head -n 1", 'glibc_ver': "ldd --version | tail -n+1 | head -n 1 | awk '{print $NF}'", 'libstdcpp_ver': "strings $(/sbin/ldconfig -p | grep libstdc++ | head -n 1 | awk '{print $NF}') | grep LIBCXX | tail -2 | head -n 1", 'cpu_isa': 'cat /proc/cpuinfo | grep flags | head -n 1'}
cmds_all = {'linux': cmds_linux}
PLATFORM = None
GPU_TYPE = None
PATH_TO_DIR = 'tensorflow/tools/tensorflow_builder/config_detector'

def run_shell_cmd(args):
    if False:
        for i in range(10):
            print('nop')
    'Executes shell commands and returns output.\n\n  Args:\n    args: String of shell commands to run.\n\n  Returns:\n    Tuple output (stdoutdata, stderrdata) from running the shell commands.\n  '
    proc = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.communicate()

def get_platform():
    if False:
        i = 10
        return i + 15
    "Retrieves platform information.\n\n  Currently the script only support linux. If other platoforms such as Windows\n  or MacOS is detected, it throws an error and terminates.\n\n  Returns:\n    String that is platform type.\n      e.g. 'linux'\n  "
    global PLATFORM
    cmd = 'uname'
    (out, err) = run_shell_cmd(cmd)
    platform_detected = out.strip().lower()
    if platform_detected != 'linux':
        if err and FLAGS.debug:
            print('Error in detecting platform:\n %s' % str(err))
        print('Error: Detected unsupported operating system.\nStopping...')
        sys.exit(1)
    else:
        PLATFORM = platform_detected
    return PLATFORM

def get_cpu_type():
    if False:
        for i in range(10):
            print('nop')
    "Retrieves CPU (type) information.\n\n  Returns:\n    String that is name of the CPU.\n      e.g. 'GenuineIntel'\n  "
    key = 'cpu_type'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM][key])
    cpu_detected = out.split(b':')[1].strip()
    if err and FLAGS.debug:
        print('Error in detecting CPU type:\n %s' % str(err))
    return cpu_detected

def get_cpu_arch():
    if False:
        i = 10
        return i + 15
    "Retrieves processor architecture type (32-bit or 64-bit).\n\n  Returns:\n    String that is CPU architecture.\n      e.g. 'x86_64'\n  "
    key = 'cpu_arch'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM][key])
    if err and FLAGS.debug:
        print('Error in detecting CPU arch:\n %s' % str(err))
    return out.strip(b'\n')

def get_distrib():
    if False:
        for i in range(10):
            print('nop')
    "Retrieves distribution name of the operating system.\n\n  Returns:\n    String that is the name of distribution.\n      e.g. 'Ubuntu'\n  "
    key = 'distrib'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM][key])
    if err and FLAGS.debug:
        print('Error in detecting distribution:\n %s' % str(err))
    return out.strip(b'\n')

def get_distrib_version():
    if False:
        i = 10
        return i + 15
    "Retrieves distribution version of the operating system.\n\n  Returns:\n    String that is the distribution version.\n      e.g. '14.04'\n  "
    key = 'distrib_ver'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM][key])
    if err and FLAGS.debug:
        print('Error in detecting distribution version:\n %s' % str(err))
    return out.strip(b'\n')

def get_gpu_type():
    if False:
        return 10
    "Retrieves GPU type.\n\n  Returns:\n    String that is the name of the detected NVIDIA GPU.\n      e.g. 'Tesla K80'\n\n    'unknown' will be returned if detected GPU type is an unknown name.\n      Unknown name refers to any GPU name that is not specified in this page:\n      https://developer.nvidia.com/cuda-gpus\n  "
    global GPU_TYPE
    key = 'gpu_type_no_sudo'
    gpu_dict = cuda_compute_capability.retrieve_from_golden()
    (out, err) = run_shell_cmd(cmds_all[PLATFORM][key])
    ret_val = out.split(b' ')
    gpu_id = ret_val[0]
    if err and FLAGS.debug:
        print('Error in detecting GPU type:\n %s' % str(err))
    if not isinstance(ret_val, list):
        GPU_TYPE = 'unknown'
        return (gpu_id, GPU_TYPE)
    else:
        if '[' or ']' in ret_val[1]:
            gpu_release = ret_val[1].replace(b'[', b'') + b' '
            gpu_release += ret_val[2].replace(b']', b'').strip(b'\n')
        else:
            gpu_release = ret_val[1].replace('\n', ' ')
        if gpu_release not in gpu_dict:
            GPU_TYPE = 'unknown'
        else:
            GPU_TYPE = gpu_release
        return (gpu_id, GPU_TYPE)

def get_gpu_count():
    if False:
        for i in range(10):
            print('nop')
    "Retrieves total number of GPU's available in the system.\n\n  Returns:\n    Integer that is the total # of GPU's found.\n  "
    key = 'gpu_count_no_sudo'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM][key])
    if err and FLAGS.debug:
        print('Error in detecting GPU count:\n %s' % str(err))
    return out.strip(b'\n')

def get_cuda_version_all():
    if False:
        i = 10
        return i + 15
    "Retrieves all additional CUDA versions available (other than default).\n\n  For retrieving default CUDA version, use `get_cuda_version` function.\n\n  stderr is silenced by default. Setting FLAGS.debug mode will not enable it.\n  Remove `2> /dev/null` command from `cmds_linux['cuda_ver_dflt']` to enable\n  stderr.\n\n  Returns:\n    List of all CUDA versions found (except default version).\n      e.g. ['10.1', '10.2']\n  "
    key = 'cuda_ver_all'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
    ret_val = out.split(b'\n')
    filtered = []
    for item in ret_val:
        if item not in ['\n', '']:
            filtered.append(item)
    all_vers = []
    for item in filtered:
        ver_re = re.search('.*/cuda(\\-[\\d]+\\.[\\d]+)?', item.decode('utf-8'))
        if ver_re.group(1):
            all_vers.append(ver_re.group(1).strip('-'))
    if err and FLAGS.debug:
        print('Error in detecting CUDA version:\n %s' % str(err))
    return all_vers

def get_cuda_version_default():
    if False:
        print('Hello World!')
    "Retrieves default CUDA version.\n\n  Default version is the version found in `/usr/local/cuda/` installation.\n\n  stderr is silenced by default. Setting FLAGS.debug mode will not enable it.\n  Remove `2> /dev/null` command from `cmds_linux['cuda_ver_dflt']` to enable\n  stderr.\n\n  It iterates through two types of version retrieval method:\n    1) Using `nvcc`: If `nvcc` is not available, then it uses next method.\n    2) Read version file (`version.txt`) found in CUDA install directory.\n\n  Returns:\n    String that is the default CUDA version.\n      e.g. '10.1'\n  "
    key = 'cuda_ver_dflt'
    out = ''
    cmd_list = cmds_all[PLATFORM.lower()][key]
    for (i, cmd) in enumerate(cmd_list):
        try:
            (out, err) = run_shell_cmd(cmd)
            if not out:
                raise Exception(err)
        except Exception as e:
            if FLAGS.debug:
                print('\nWarning: Encountered issue while retrieving default CUDA version. (%s) Trying a different method...\n' % e)
            if i == len(cmd_list) - 1:
                if FLAGS.debug:
                    print('Error: Cannot retrieve CUDA default version.\nStopping...')
            else:
                pass
    return out.strip('\n')

def get_cuda_compute_capability(source_from_url=False):
    if False:
        while True:
            i = 10
    "Retrieves CUDA compute capability based on the detected GPU type.\n\n  This function uses the `cuda_compute_capability` module to retrieve the\n  corresponding CUDA compute capability for the given GPU type.\n\n  Args:\n    source_from_url: Boolean deciding whether to source compute capability\n                     from NVIDIA website or from a local golden file.\n\n  Returns:\n    List of all supported CUDA compute capabilities for the given GPU type.\n      e.g. ['3.5', '3.7']\n  "
    if not GPU_TYPE:
        if FLAGS.debug:
            print('Warning: GPU_TYPE is empty. Make sure to call `get_gpu_type()` first.')
    elif GPU_TYPE == 'unknown':
        if FLAGS.debug:
            print('Warning: Unknown GPU is detected. Skipping CUDA compute capability retrieval.')
    else:
        if source_from_url:
            cuda_compute_capa = cuda_compute_capability.retrieve_from_web()
        else:
            cuda_compute_capa = cuda_compute_capability.retrieve_from_golden()
        return cuda_compute_capa[GPU_TYPE]
    return

def get_cudnn_version():
    if False:
        print('Hello World!')
    "Retrieves the version of cuDNN library detected.\n\n  Returns:\n    String that is the version of cuDNN library detected.\n      e.g. '7.5.0'\n  "
    key = 'cudnn_ver'
    cmds = cmds_all[PLATFORM.lower()][key]
    (out, err) = run_shell_cmd(cmds[0])
    if err and FLAGS.debug:
        print('Error in finding `cudnn.h`:\n %s' % str(err))
    if len(out.split(b' ')) > 1:
        cmd = cmds[0] + ' | ' + cmds[1]
        (out_re, err_re) = run_shell_cmd(cmd)
        if err_re and FLAGS.debug:
            print('Error in detecting cuDNN version:\n %s' % str(err_re))
        return out_re.strip(b'\n')
    else:
        return

def get_gcc_version():
    if False:
        i = 10
        return i + 15
    "Retrieves version of GCC detected.\n\n  Returns:\n    String that is the version of GCC.\n      e.g. '7.3.0'\n  "
    key = 'gcc_ver'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
    if err and FLAGS.debug:
        print('Error in detecting GCC version:\n %s' % str(err))
    return out.strip(b'\n')

def get_glibc_version():
    if False:
        print('Hello World!')
    "Retrieves version of GLIBC detected.\n\n  Returns:\n    String that is the version of GLIBC.\n      e.g. '2.24'\n  "
    key = 'glibc_ver'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
    if err and FLAGS.debug:
        print('Error in detecting GCC version:\n %s' % str(err))
    return out.strip(b'\n')

def get_libstdcpp_version():
    if False:
        for i in range(10):
            print('nop')
    "Retrieves version of libstdc++ detected.\n\n  Returns:\n    String that is the version of libstdc++.\n      e.g. '3.4.25'\n  "
    key = 'libstdcpp_ver'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
    if err and FLAGS.debug:
        print('Error in detecting libstdc++ version:\n %s' % str(err))
    ver = out.split(b'_')[-1].replace(b'\n', b'')
    return ver

def get_cpu_isa_version():
    if False:
        for i in range(10):
            print('nop')
    "Retrieves all Instruction Set Architecture(ISA) available.\n\n  Required ISA(s): 'avx', 'avx2', 'avx512f', 'sse4', 'sse4_1'\n\n  Returns:\n    Tuple\n      (list of available ISA, list of missing ISA)\n  "
    key = 'cpu_isa'
    (out, err) = run_shell_cmd(cmds_all[PLATFORM.lower()][key])
    if err and FLAGS.debug:
        print('Error in detecting supported ISA:\n %s' % str(err))
    ret_val = out
    required_isa = ['avx', 'avx2', 'avx512f', 'sse4', 'sse4_1']
    found = []
    missing = []
    for isa in required_isa:
        for sys_isa in ret_val.split(b' '):
            if isa == sys_isa:
                if isa not in found:
                    found.append(isa)
    missing = list(set(required_isa) - set(found))
    return (found, missing)

def get_python_version():
    if False:
        print('Hello World!')
    "Retrieves default Python version.\n\n  Returns:\n    String that is the version of default Python.\n      e.g. '2.7.4'\n  "
    ver = str(sys.version_info)
    mmm = re.search('.*major=([\\d]), minor=([\\d]), micro=([\\d]+),.*', ver)
    return mmm.group(1) + '.' + mmm.group(2) + '.' + mmm.group(3)

def get_all_configs():
    if False:
        i = 10
        return i + 15
    'Runs all functions for detecting user machine configurations.\n\n  Returns:\n    Tuple\n      (List of all configurations found,\n       List of all missing configurations,\n       List of all configurations found with warnings,\n       Dict of all configurations)\n  '
    all_functions = collections.OrderedDict([('Platform', get_platform()), ('CPU', get_cpu_type()), ('CPU arch', get_cpu_arch()), ('Distribution', get_distrib()), ('Distribution version', get_distrib_version()), ('GPU', get_gpu_type()[1]), ('GPU count', get_gpu_count()), ('CUDA version (default)', get_cuda_version_default()), ('CUDA versions (all)', get_cuda_version_all()), ('CUDA compute capability', get_cuda_compute_capability(get_gpu_type()[1])), ('cuDNN version', get_cudnn_version()), ('GCC version', get_gcc_version()), ('Python version (default)', get_python_version()), ('GNU C Lib (glibc) version', get_glibc_version()), ('libstdc++ version', get_libstdcpp_version()), ('CPU ISA (min requirement)', get_cpu_isa_version())])
    configs_found = []
    json_data = {}
    missing = []
    warning = []
    for (config, call_func) in all_functions.items():
        ret_val = call_func
        if not ret_val:
            configs_found.append([config, '\x1b[91m\x1b[1mMissing\x1b[0m'])
            missing.append([config])
            json_data[config] = ''
        elif ret_val == 'unknown':
            configs_found.append([config, '\x1b[93m\x1b[1mUnknown type\x1b[0m'])
            warning.append([config, ret_val])
            json_data[config] = 'unknown'
        elif 'ISA' in config:
            if not ret_val[1]:
                configs_found.append([config, ret_val[0]])
                json_data[config] = ret_val[0]
            else:
                configs_found.append([config, '\x1b[91m\x1b[1mMissing ' + str(ret_val[1][1:-1]) + '\x1b[0m'])
                missing.append([config, '\n\t=> Found %s but missing %s' % (str(ret_val[0]), str(ret_val[1]))])
                json_data[config] = ret_val[0]
        else:
            configs_found.append([config, ret_val])
            json_data[config] = ret_val
    return (configs_found, missing, warning, json_data)

def print_all_configs(configs, missing, warning):
    if False:
        while True:
            i = 10
    'Prints the status and info on all configurations in a table format.\n\n  Args:\n    configs: List of all configurations found.\n    missing: List of all configurations that are missing.\n    warning: List of all configurations found with warnings.\n  '
    print_text = ''
    llen = 65
    for (i, row) in enumerate(configs):
        if i != 0:
            print_text += '-' * llen + '\n'
        if isinstance(row[1], list):
            val = ', '.join(row[1])
        else:
            val = row[1]
        print_text += ' {: <28}'.format(row[0]) + '    {: <25}'.format(val) + '\n'
    print_text += '=' * llen
    print('\n\n {: ^32}    {: ^25}'.format('Configuration(s)', 'Detected value(s)'))
    print('=' * llen)
    print(print_text)
    if missing:
        print('\n * ERROR: The following configurations are missing:')
        for m in missing:
            print('   ', *m)
    if warning:
        print('\n * WARNING: The following configurations could cause issues:')
        for w in warning:
            print('   ', *w)
    if not missing and (not warning):
        print('\n * INFO: Successfully found all configurations.')
    print('\n')

def save_to_file(json_data, filename):
    if False:
        print('Hello World!')
    'Saves all detected configuration(s) into a JSON file.\n\n  Args:\n    json_data: Dict of all configurations found.\n    filename: String that is the name of the output JSON file.\n  '
    if filename[-5:] != '.json':
        print('filename: %s' % filename)
        filename += '.json'
    with open(PATH_TO_DIR + '/' + filename, 'w') as f:
        json.dump(json_data, f, sort_keys=True, indent=4)
    print(' Successfully wrote configs to file `%s`.\n' % filename)

def manage_all_configs(save_results, filename):
    if False:
        for i in range(10):
            print('nop')
    'Manages configuration detection and retrieval based on user input.\n\n  Args:\n    save_results: Boolean indicating whether to save the results to a file.\n    filename: String that is the name of the output JSON file.\n  '
    all_configs = get_all_configs()
    print_all_configs(all_configs[0], all_configs[1], all_configs[2])
    if save_results:
        save_to_file(all_configs[3], filename)

def main(argv):
    if False:
        while True:
            i = 10
    if len(argv) > 3:
        raise app.UsageError('Too many command-line arguments.')
    manage_all_configs(save_results=FLAGS.save_output, filename=FLAGS.filename)
if __name__ == '__main__':
    app.run(main)