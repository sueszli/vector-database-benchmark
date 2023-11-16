import platform
import subprocess
import pkg_resources
import psutil

def get_python_version():
    if False:
        i = 10
        return i + 15
    return platform.python_version()

def get_pip_version():
    if False:
        for i in range(10):
            print('nop')
    try:
        pip_version = subprocess.check_output(['pip', '--version']).decode().split()[1]
    except Exception as e:
        pip_version = str(e)
    return pip_version

def get_oi_version():
    if False:
        while True:
            i = 10
    try:
        oi_version_cmd = subprocess.check_output(['interpreter', '--version']).decode().split()[1]
    except Exception as e:
        oi_version_cmd = str(e)
    oi_version_pkg = pkg_resources.get_distribution('open-interpreter').version
    oi_version = (oi_version_cmd, oi_version_pkg)
    return oi_version

def get_os_version():
    if False:
        i = 10
        return i + 15
    return platform.platform()

def get_cpu_info():
    if False:
        return 10
    return platform.processor()

def get_ram_info():
    if False:
        return 10
    vm = psutil.virtual_memory()
    used_ram_gb = vm.used / 1024 ** 3
    free_ram_gb = vm.free / 1024 ** 3
    total_ram_gb = vm.total / 1024 ** 3
    return f'{total_ram_gb:.2f} GB, used: {used_ram_gb:.2f}, free: {free_ram_gb:.2f}'

def interpreter_info(interpreter):
    if False:
        return 10
    try:
        if interpreter.local:
            try:
                curl = subprocess.check_output(f'curl {interpreter.api_base}')
            except Exception as e:
                curl = str(e)
        else:
            curl = 'Not local'
        return f'\n\n        Interpreter Info\n        Vision: {interpreter.vision}\n        Model: {interpreter.model}\n        Function calling: {interpreter.function_calling_llm}\n        Context window: {interpreter.context_window}\n        Max tokens: {interpreter.max_tokens}\n\n        Auto run: {interpreter.auto_run}\n        API base: {interpreter.api_base}\n        Local: {interpreter.local}\n\n        Curl output: {curl}\n    '
    except:
        return "Error, couldn't get interpreter info"

def system_info(interpreter):
    if False:
        return 10
    oi_version = get_oi_version()
    print(f'\n        Python Version: {get_python_version()}\n        Pip Version: {get_pip_version()}\n        Open-interpreter Version: cmd:{oi_version[0]}, pkg: {oi_version[1]}\n        OS Version and Architecture: {get_os_version()}\n        CPU Info: {get_cpu_info()}\n        RAM Info: {get_ram_info()}\n        {interpreter_info(interpreter)}\n    ')