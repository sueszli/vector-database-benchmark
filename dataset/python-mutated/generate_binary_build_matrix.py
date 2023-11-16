"""Generates a matrix to be utilized through github actions

Will output a condensed version of the matrix if on a pull request that only
includes the latest version of python we support built on three different
architectures:
    * CPU
    * Latest CUDA
    * Latest ROCM
"""
from typing import Dict, List, Optional, Tuple
CUDA_ARCHES = ['11.8', '12.1']
ROCM_ARCHES = ['5.6', '5.7']
CPU_CXX11_ABI_ARCH = ['cpu-cxx11-abi']
CPU_AARCH64_ARCH = ['cpu-aarch64']
PYTORCH_EXTRA_INSTALL_REQUIREMENTS = "nvidia-cuda-nvrtc-cu12==12.1.105; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cuda-runtime-cu12==12.1.105; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cuda-cupti-cu12==12.1.105; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cudnn-cu12==8.9.2.26; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cublas-cu12==12.1.3.1; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cufft-cu12==11.0.2.54; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-curand-cu12==10.3.2.106; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cusolver-cu12==11.4.5.107; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-cusparse-cu12==12.1.0.106; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-nccl-cu12==2.19.3; platform_system == 'Linux' and platform_machine == 'x86_64' | nvidia-nvtx-cu12==12.1.105; platform_system == 'Linux' and platform_machine == 'x86_64'"

def get_nccl_submodule_version() -> str:
    if False:
        for i in range(10):
            print('nop')
    from pathlib import Path
    nccl_version_mk = Path(__file__).absolute().parent.parent.parent / 'third_party' / 'nccl' / 'nccl' / 'makefiles' / 'version.mk'
    if not nccl_version_mk.exists():
        raise RuntimeError('Please make sure that nccl submodule is checked out when importing this script')
    with nccl_version_mk.open('r') as f:
        content = f.read()
    d = {}
    for l in content.split('\n'):
        if not l.startswith('NCCL_'):
            continue
        (k, v) = l.split(':=')
        d[k.strip()] = v.strip()
    return f"{d['NCCL_MAJOR']}.{d['NCCL_MINOR']}.{d['NCCL_PATCH']}"

def get_nccl_wheel_version() -> str:
    if False:
        while True:
            i = 10
    import re
    requrements = map(str.strip, re.split('[;|]', PYTORCH_EXTRA_INSTALL_REQUIREMENTS))
    return [x for x in requrements if x.startswith('nvidia-nccl-cu')][0].split('==')[1]

def validate_nccl_dep_consistency() -> None:
    if False:
        for i in range(10):
            print('nop')
    wheel_ver = get_nccl_wheel_version()
    submodule_ver = get_nccl_submodule_version()
    if wheel_ver != submodule_ver:
        raise RuntimeError(f'NCCL submodule version {submodule_ver} differs from wheel version {wheel_ver}')

def arch_type(arch_version: str) -> str:
    if False:
        print('Hello World!')
    if arch_version in CUDA_ARCHES:
        return 'cuda'
    elif arch_version in ROCM_ARCHES:
        return 'rocm'
    elif arch_version in CPU_CXX11_ABI_ARCH:
        return 'cpu-cxx11-abi'
    elif arch_version in CPU_AARCH64_ARCH:
        return 'cpu-aarch64'
    else:
        return 'cpu'
DEFAULT_TAG = 'main'
WHEEL_CONTAINER_IMAGES = {**{gpu_arch: f'pytorch/manylinux-builder:cuda{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in CUDA_ARCHES}, **{gpu_arch: f'pytorch/manylinux-builder:rocm{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in ROCM_ARCHES}, 'cpu': f'pytorch/manylinux-builder:cpu-{DEFAULT_TAG}', 'cpu-cxx11-abi': f'pytorch/manylinuxcxx11-abi-builder:cpu-cxx11-abi-{DEFAULT_TAG}', 'cpu-aarch64': f'pytorch/manylinuxaarch64-builder:cpu-aarch64-{DEFAULT_TAG}'}
CONDA_CONTAINER_IMAGES = {**{gpu_arch: f'pytorch/conda-builder:cuda{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in CUDA_ARCHES}, 'cpu': f'pytorch/conda-builder:cpu-{DEFAULT_TAG}'}
PRE_CXX11_ABI = 'pre-cxx11'
CXX11_ABI = 'cxx11-abi'
RELEASE = 'release'
DEBUG = 'debug'
LIBTORCH_CONTAINER_IMAGES: Dict[Tuple[str, str], str] = {**{(gpu_arch, PRE_CXX11_ABI): f'pytorch/manylinux-builder:cuda{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in CUDA_ARCHES}, **{(gpu_arch, CXX11_ABI): f'pytorch/libtorch-cxx11-builder:cuda{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in CUDA_ARCHES}, **{(gpu_arch, PRE_CXX11_ABI): f'pytorch/manylinux-builder:rocm{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in ROCM_ARCHES}, **{(gpu_arch, CXX11_ABI): f'pytorch/libtorch-cxx11-builder:rocm{gpu_arch}-{DEFAULT_TAG}' for gpu_arch in ROCM_ARCHES}, ('cpu', PRE_CXX11_ABI): f'pytorch/manylinux-builder:cpu-{DEFAULT_TAG}', ('cpu', CXX11_ABI): f'pytorch/libtorch-cxx11-builder:cpu-{DEFAULT_TAG}'}
FULL_PYTHON_VERSIONS = ['3.8', '3.9', '3.10', '3.11']

def translate_desired_cuda(gpu_arch_type: str, gpu_arch_version: str) -> str:
    if False:
        return 10
    return {'cpu': 'cpu', 'cpu-aarch64': 'cpu', 'cpu-cxx11-abi': 'cpu-cxx11-abi', 'cuda': f"cu{gpu_arch_version.replace('.', '')}", 'rocm': f'rocm{gpu_arch_version}'}.get(gpu_arch_type, gpu_arch_version)

def list_without(in_list: List[str], without: List[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    return [item for item in in_list if item not in without]

def generate_conda_matrix(os: str) -> List[Dict[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    ret: List[Dict[str, str]] = []
    arches = ['cpu']
    python_versions = FULL_PYTHON_VERSIONS
    if os == 'linux' or os == 'windows':
        arches += CUDA_ARCHES
    for python_version in python_versions:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = '' if arch_version == 'cpu' else arch_version
            ret.append({'python_version': python_version, 'gpu_arch_type': gpu_arch_type, 'gpu_arch_version': gpu_arch_version, 'desired_cuda': translate_desired_cuda(gpu_arch_type, gpu_arch_version), 'container_image': CONDA_CONTAINER_IMAGES[arch_version], 'package_type': 'conda', 'build_name': f'conda-py{python_version}-{gpu_arch_type}{gpu_arch_version}'.replace('.', '_')})
    return ret

def generate_libtorch_matrix(os: str, abi_version: str, arches: Optional[List[str]]=None, libtorch_variants: Optional[List[str]]=None) -> List[Dict[str, str]]:
    if False:
        while True:
            i = 10
    if arches is None:
        arches = ['cpu']
        if os == 'linux':
            arches += CUDA_ARCHES
            arches += ROCM_ARCHES
        elif os == 'windows':
            arches += CUDA_ARCHES
    if libtorch_variants is None:
        libtorch_variants = ['shared-with-deps', 'shared-without-deps', 'static-with-deps', 'static-without-deps']
    ret: List[Dict[str, str]] = []
    for arch_version in arches:
        for libtorch_variant in libtorch_variants:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = '' if arch_version == 'cpu' else arch_version
            if gpu_arch_type == 'rocm' and 'without-deps' in libtorch_variant:
                continue
            ret.append({'gpu_arch_type': gpu_arch_type, 'gpu_arch_version': gpu_arch_version, 'desired_cuda': translate_desired_cuda(gpu_arch_type, gpu_arch_version), 'libtorch_variant': libtorch_variant, 'libtorch_config': abi_version if os == 'windows' else '', 'devtoolset': abi_version if os != 'windows' else '', 'container_image': LIBTORCH_CONTAINER_IMAGES[arch_version, abi_version] if os != 'windows' else '', 'package_type': 'libtorch', 'build_name': f'libtorch-{gpu_arch_type}{gpu_arch_version}-{libtorch_variant}-{abi_version}'.replace('.', '_')})
    return ret

def generate_wheels_matrix(os: str, arches: Optional[List[str]]=None, python_versions: Optional[List[str]]=None, gen_special_an_non_special_wheel: bool=True) -> List[Dict[str, str]]:
    if False:
        while True:
            i = 10
    package_type = 'wheel'
    if os == 'linux' or os == 'linux-aarch64':
        package_type = 'manywheel'
    if python_versions is None:
        python_versions = FULL_PYTHON_VERSIONS
    if arches is None:
        arches = ['cpu']
        if os == 'linux':
            arches += CPU_CXX11_ABI_ARCH + CUDA_ARCHES + ROCM_ARCHES
        elif os == 'windows':
            arches += CUDA_ARCHES
        elif os == 'linux-aarch64':
            arches = ['cpu-aarch64']
    ret: List[Dict[str, str]] = []
    for python_version in python_versions:
        for arch_version in arches:
            gpu_arch_type = arch_type(arch_version)
            gpu_arch_version = '' if arch_version == 'cpu' or arch_version == 'cpu-cxx11-abi' or arch_version == 'cpu-aarch64' else arch_version
            if arch_version == '12.1' and os == 'linux':
                ret.append({'python_version': python_version, 'gpu_arch_type': gpu_arch_type, 'gpu_arch_version': gpu_arch_version, 'desired_cuda': translate_desired_cuda(gpu_arch_type, gpu_arch_version), 'devtoolset': '', 'container_image': WHEEL_CONTAINER_IMAGES[arch_version], 'package_type': package_type, 'pytorch_extra_install_requirements': PYTORCH_EXTRA_INSTALL_REQUIREMENTS, 'build_name': f'{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}-with-pypi-cudnn'.replace('.', '_')})
                if not gen_special_an_non_special_wheel:
                    continue
            ret.append({'python_version': python_version, 'gpu_arch_type': gpu_arch_type, 'gpu_arch_version': gpu_arch_version, 'desired_cuda': translate_desired_cuda(gpu_arch_type, gpu_arch_version), 'devtoolset': 'cxx11-abi' if arch_version == 'cpu-cxx11-abi' else '', 'container_image': WHEEL_CONTAINER_IMAGES[arch_version], 'package_type': package_type, 'build_name': f'{package_type}-py{python_version}-{gpu_arch_type}{gpu_arch_version}'.replace('.', '_'), 'pytorch_extra_install_requirements': PYTORCH_EXTRA_INSTALL_REQUIREMENTS if os != 'linux' else ''})
    return ret
validate_nccl_dep_consistency()