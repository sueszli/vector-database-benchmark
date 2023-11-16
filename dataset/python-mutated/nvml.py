"""Imports pynvml with sanity checks and custom patches."""
from typing import Tuple
import functools
import os
import sys
import textwrap
import warnings
ALLOW_LEGACY_PYNVML = os.getenv('ALLOW_LEGACY_PYNVML', '')
ALLOW_LEGACY_PYNVML = ALLOW_LEGACY_PYNVML.lower() not in ('false', '0', '')
try:
    import pynvml
    if not (hasattr(pynvml, 'NVML_BRAND_NVIDIA_RTX') or hasattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses_v2')) and (not ALLOW_LEGACY_PYNVML):
        raise ImportError('pynvml library is outdated.')
    if not hasattr(pynvml, '_nvmlGetFunctionPointer'):
        import pynvml.nvml as pynvml
except (ImportError, SyntaxError, RuntimeError) as e:
    _pynvml = sys.modules.get('pynvml', None)
    raise ImportError(textwrap.dedent('        pynvml is missing or an outdated version is installed.\n\n        We require nvidia-ml-py>=11.450.129, and the official NVIDIA python bindings\n        should be used; neither nvidia-ml-py3 nor gpuopenanalytics/pynvml.\n        For more details, please refer to: https://github.com/wookayin/gpustat/issues/107\n\n        The root cause: ' + str(e) + '\n\n        Your pynvml installation: ' + repr(_pynvml) + "\n\n        -----------------------------------------------------------\n        Please reinstall `gpustat`:\n\n        $ pip install --force-reinstall gpustat\n\n        If it still does not fix the problem, please uninstall pynvml packages and reinstall nvidia-ml-py manually:\n\n        $ pip uninstall nvidia-ml-py3 pynvml\n        $ pip install --force-reinstall --ignore-installed 'nvidia-ml-py'\n        ")) from e

class NvidiaCompatibilityWarning(UserWarning):
    pass

def check_driver_nvml_version(driver_version_str: str):
    if False:
        while True:
            i = 10
    'Show warnings when an incompatible driver is used.'

    def safeint(v) -> int:
        if False:
            return 10
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0
    driver_version = tuple((safeint(v) for v in driver_version_str.strip().split('.')))
    is_pynvml_535_77 = hasattr(pynvml.c_nvmlProcessInfo_t, 'usedGpuCcProtectedMemory') and pynvml.c_nvmlProcessInfo_t.__name__ == 'c_nvmlProcessInfo_t'
    if (535, 43) <= driver_version < (535, 86):
        if not is_pynvml_535_77:
            warnings.warn(f'This version of NVIDIA Driver {driver_version_str} is incompatible, process information will be inaccurate. Upgrade the NVIDIA driver to 535.104.05 or higher, or use nvidia-ml-py==12.535.77. For more details, see https://github.com/wookayin/gpustat/issues/161.', category=NvidiaCompatibilityWarning, stacklevel=2)
    elif is_pynvml_535_77:
        warnings.warn('This version of nvidia-ml-py (possibly 12.535.77) is incompatible. Please upgrade nvidia-ml-py to the latest version. (pip install --upgrade --force-reinstall nvidia-ml-py)', category=NvidiaCompatibilityWarning, stacklevel=2)
_original_nvmlGetFunctionPointer = pynvml._nvmlGetFunctionPointer
_original_nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo

class pynvml_monkeypatch:

    @staticmethod
    def original_nvmlGetFunctionPointer(name):
        if False:
            for i in range(10):
                print('nop')
        return _original_nvmlGetFunctionPointer(name)
    FUNCTION_FALLBACKS = {'nvmlDeviceGetComputeRunningProcesses_v3': 'nvmlDeviceGetComputeRunningProcesses_v2', 'nvmlDeviceGetGraphicsRunningProcesses_v3': 'nvmlDeviceGetGraphicsRunningProcesses_v2'}

    @staticmethod
    @functools.wraps(pynvml._nvmlGetFunctionPointer)
    def _nvmlGetFunctionPointer(name):
        if False:
            print('Hello World!')
        'Our monkey-patched pynvml._nvmlGetFunctionPointer().\n\n        See also:\n            test_gpustat::NvidiaDriverMock for test scenarios. See #107.\n        '
        M = pynvml_monkeypatch
        try:
            ret = M.original_nvmlGetFunctionPointer(name)
            return ret
        except pynvml.NVMLError_FunctionNotFound:
            if name in M.FUNCTION_FALLBACKS:
                ret = M.original_nvmlGetFunctionPointer(M.FUNCTION_FALLBACKS[name])
                pynvml._nvmlGetFunctionPointer_cache[name] = ret
            else:
                raise
        return ret

    @staticmethod
    def original_nvmlDeviceGetMemoryInfo(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return _original_nvmlDeviceGetMemoryInfo(*args, **kwargs)
    has_memoryinfo_v2 = None

    @staticmethod
    @functools.wraps(pynvml.nvmlDeviceGetMemoryInfo)
    def nvmlDeviceGetMemoryInfo(handle):
        if False:
            i = 10
            return i + 15
        'A patched version of nvmlDeviceGetMemoryInfo.\n\n        This tries `version=N.nvmlMemory_v2` if the nvmlDeviceGetMemoryInfo_v2\n        function is available (for driver >= 515), or fallback to the legacy\n        v1 API for (driver < 515) to yield a correct result. See #141.\n        '
        M = pynvml_monkeypatch
        if M.has_memoryinfo_v2 is None:
            try:
                pynvml._nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo_v2')
                M.has_memoryinfo_v2 = True
            except pynvml.NVMLError_FunctionNotFound:
                M.has_memoryinfo_v2 = False
        if hasattr(pynvml, 'nvmlMemory_v2'):
            try:
                memory = M.original_nvmlDeviceGetMemoryInfo(handle, version=pynvml.nvmlMemory_v2)
            except pynvml.NVMLError_FunctionNotFound:
                memory = M.original_nvmlDeviceGetMemoryInfo(handle)
        else:
            if M.has_memoryinfo_v2:
                warnings.warn('Your NVIDIA driver requires a compatible version of pynvml (>= 11.510.69) installed to display the correct memory usage information (See #141 for more details). Please try `pip install --upgrade nvidia-ml-py`.', category=UserWarning)
            memory = M.original_nvmlDeviceGetMemoryInfo(handle)
        return memory
setattr(pynvml, '_nvmlGetFunctionPointer', pynvml_monkeypatch._nvmlGetFunctionPointer)
setattr(pynvml, 'nvmlDeviceGetMemoryInfo', pynvml_monkeypatch.nvmlDeviceGetMemoryInfo)
__all__ = ['pynvml']