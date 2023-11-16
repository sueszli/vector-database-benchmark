import math
import os
import subprocess
import sys
import warnings
NO_DRIVER = (math.inf, math.inf)
NUMBA_CHECK_VERSION_CMD = "from ctypes import c_int, byref\nfrom numba import cuda\ndv = c_int(0)\ncuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))\ndrv_major = dv.value // 1000\ndrv_minor = (dv.value - (drv_major * 1000)) // 10\nrun_major, run_minor = cuda.runtime.get_version()\nprint(f'{drv_major} {drv_minor} {run_major} {run_minor}')\n"

def check_disabled_in_env():
    if False:
        for i in range(10):
            print('nop')
    check = os.getenv('PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED')
    if check is not None:
        try:
            check = int(check)
        except ValueError:
            check = False
    else:
        check = True
    return not check

def get_versions():
    if False:
        i = 10
        return i + 15
    cp = subprocess.run([sys.executable, '-c', NUMBA_CHECK_VERSION_CMD], capture_output=True)
    if cp.returncode:
        msg = f'Error getting driver and runtime versions:\n\nstdout:\n\n{cp.stdout.decode()}\n\nstderr:\n\n{cp.stderr.decode()}\n\nNot patching Numba'
        warnings.warn(msg, UserWarning)
        return NO_DRIVER
    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])
    runtime_version = tuple(versions[2:])
    return (driver_version, runtime_version)

def safe_get_versions():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a 2-tuple of deduced driver and runtime versions.\n\n    To ensure that this function does not initialize a CUDA context,\n    calls to the runtime and driver are made in a subprocess.\n\n    If PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED is set\n    in the environment, then this subprocess call is not launched.\n    To specify the driver and runtime versions of the environment\n    in this case, set PTXCOMPILER_KNOWN_DRIVER_VERSION and\n    PTXCOMPILER_KNOWN_RUNTIME_VERSION appropriately.\n    '
    if check_disabled_in_env():
        try:
            driver_version = os.environ['PTXCOMPILER_KNOWN_DRIVER_VERSION'].split('.')
            runtime_version = os.environ['PTXCOMPILER_KNOWN_RUNTIME_VERSION'].split('.')
            (driver_version, runtime_version) = (tuple(map(int, driver_version)), tuple(map(int, runtime_version)))
        except (KeyError, ValueError):
            warnings.warn('No way to determine driver and runtime versions for patching, set PTXCOMPILER_KNOWN_DRIVER_VERSION and PTXCOMPILER_KNOWN_RUNTIME_VERSION')
            return NO_DRIVER
    else:
        (driver_version, runtime_version) = get_versions()
    return (driver_version, runtime_version)