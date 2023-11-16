import platform
import subprocess
import sys
import distro
envs_template = '\nPaddle version: {paddle_version}\nPaddle With CUDA: {paddle_with_cuda}\n\nOS: {os_info}\nGCC version: {gcc_version}\nClang version: {clang_version}\nCMake version: {cmake_version}\nLibc version: {libc_version}\nPython version: {python_version}\n\nCUDA version: {cuda_version}\ncuDNN version: {cudnn_version}\nNvidia driver version: {nvidia_driver_version}\nNvidia driver List: {nvidia_gpu_driver}\n'
envs = {}

def get_paddle_info():
    if False:
        for i in range(10):
            print('nop')
    try:
        import paddle
        envs['paddle_version'] = paddle.__version__
        envs['paddle_with_cuda'] = paddle.base.core.is_compiled_with_cuda()
    except:
        envs['paddle_version'] = 'N/A'
        envs['paddle_with_cuda'] = 'N/A'

def get_os_info():
    if False:
        i = 10
        return i + 15
    if platform.system() == 'Darwin':
        plat = 'macOS'
        ver = run_shell_command('sw_vers -productVersion').strip('\n')
    elif platform.system() == 'Linux':
        plat = distro.id()
        ver = distro.version()
    elif platform.system() == 'Windows':
        plat = 'Windows'
        ver = platform.win32_ver()[0]
    else:
        plat = 'N/A'
        ver = 'N/A'
    envs['os_info'] = f'{plat} {ver}'

def get_gcc_version():
    if False:
        print('Hello World!')
    try:
        envs['gcc_version'] = run_shell_command('gcc --version').split('\n')[0].split('gcc ')[1]
    except:
        envs['gcc_version'] = 'N/A'

def get_clang_version():
    if False:
        while True:
            i = 10
    try:
        envs['clang_version'] = run_shell_command('clang --version').split('\n')[0].split('clang version ')[1]
    except:
        envs['clang_version'] = 'N/A'

def get_cmake_version():
    if False:
        print('Hello World!')
    try:
        envs['cmake_version'] = run_shell_command('cmake --version').split('\n')[0].split('cmake ')[1]
    except:
        envs['cmake_version'] = 'N/A'

def get_libc_version():
    if False:
        while True:
            i = 10
    if platform.system() == 'Linux':
        envs['libc_version'] = ' '.join(platform.libc_ver())
    else:
        envs['libc_version'] = 'N/A'

def get_python_info():
    if False:
        print('Hello World!')
    envs['python_version'] = sys.version.split(' ')[0]

def run_shell_command(cmd):
    if False:
        for i in range(10):
            print('nop')
    (out, err) = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()
    if err:
        return None
    else:
        return out.decode('utf-8')

def get_cuda_info():
    if False:
        print('Hello World!')
    out = run_shell_command('nvcc --version')
    if out:
        envs['cuda_version'] = out.split('V')[-1].strip()
    else:
        envs['cuda_version'] = 'N/A'

def get_cudnn_info():
    if False:
        for i in range(10):
            print('nop')

    def _get_cudnn_ver(cmd):
        if False:
            i = 10
            return i + 15
        out = run_shell_command(cmd)
        if out:
            return out.split(' ')[-1].strip()
        else:
            return 'N/A'
    if platform.system() == 'Windows':
        cudnn_dll_path = run_shell_command('where cudnn*')
        if cudnn_dll_path:
            cudnn_header_path = cudnn_dll_path.split('bin')[0] + 'include\\cudnn.h'
            cmd = 'type "{0}" | findstr "{1}" | findstr /v "CUDNN_VERSION"'
        else:
            envs['cudnn_version'] = 'N/A'
            return
    else:
        cudnn_header_path = run_shell_command('whereis "cudnn.h" | awk \'{print $2}\'').strip('\n')
        if cudnn_header_path:
            cmd = 'cat "{0}" | grep "{1}" | grep -v "CUDNN_VERSION"'
            if _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MAJOR')):
                cudnn_header_path = run_shell_command('whereis "cudnn_version.h" | awk \'{print $2}\'').strip('\n')
        else:
            envs['cudnn_version'] = 'N/A'
            return
    major = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MAJOR'))
    minor = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_MINOR'))
    patch_level = _get_cudnn_ver(cmd.format(cudnn_header_path, 'CUDNN_PATCHLEVEL'))
    if major != 'N/A':
        envs['cudnn_version'] = f'{major}.{minor}.{patch_level}'
    else:
        envs['cudnn_version'] = 'N/A'

def get_driver_info():
    if False:
        return 10
    driver_ver = run_shell_command('nvidia-smi')
    if driver_ver:
        driver_ver = driver_ver.split('Driver Version:')[1].strip().split(' ')[0]
    else:
        driver_ver = 'N/A'
    envs['nvidia_driver_version'] = driver_ver

def get_nvidia_gpu_driver():
    if False:
        i = 10
        return i + 15
    if platform.system() != 'Windows' and platform.system() != 'Linux':
        envs['nvidia_gpu_driver'] = 'N/A'
        return
    try:
        nvidia_smi = 'nvidia-smi'
        gpu_list = run_shell_command(nvidia_smi + ' -L')
        result = '\n'
        for gpu_info in gpu_list.split('\n'):
            result += gpu_info.split(' (UUID:')[0] + '\n'
        envs['nvidia_gpu_driver'] = result
    except:
        envs['nvidia_gpu_driver'] = 'N/A'

def main():
    if False:
        for i in range(10):
            print('nop')
    get_paddle_info()
    get_os_info()
    get_gcc_version()
    get_clang_version()
    get_cmake_version()
    get_libc_version()
    get_python_info()
    get_cuda_info()
    get_cudnn_info()
    get_driver_info()
    get_nvidia_gpu_driver()
    print('*' * 40 + envs_template.format(**envs) + '*' * 40)
if __name__ == '__main__':
    main()