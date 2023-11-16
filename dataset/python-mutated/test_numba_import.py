import subprocess
import sys
import pytest
IS_CUDA_11 = False
try:
    from ptxcompiler.patch import NO_DRIVER, safe_get_versions
    versions = safe_get_versions()
    if versions != NO_DRIVER:
        (driver_version, runtime_version) = versions
        if driver_version < (12, 0):
            IS_CUDA_11 = True
except ModuleNotFoundError:
    pass
TEST_NUMBA_MVC_ENABLED = '\nimport numba.cuda\nimport cudf\nfrom cudf.utils._numba import _CUDFNumbaConfig, _patch_numba_mvc\n\n\n_patch_numba_mvc()\n\n@numba.cuda.jit\ndef test_kernel(x):\n    id = numba.cuda.grid(1)\n    if id < len(x):\n        x[id] += 1\n\ns = cudf.Series([1, 2, 3])\nwith _CUDFNumbaConfig():\n    test_kernel.forall(len(s))(s)\n'

@pytest.mark.skipif(not IS_CUDA_11, reason='Minor Version Compatibility test for CUDA 11')
def test_numba_mvc_enabled_cuda_11():
    if False:
        print('Hello World!')
    cp = subprocess.run([sys.executable, '-c', TEST_NUMBA_MVC_ENABLED], capture_output=True, cwd='/')
    assert cp.returncode == 0