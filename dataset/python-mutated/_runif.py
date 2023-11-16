from typing import Dict, List, Optional, Tuple
from lightning_utilities.core.imports import RequirementCache
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.testing import _runif_reasons as FabricRunIf
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.core.module import _ONNX_AVAILABLE
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
_SKLEARN_AVAILABLE = RequirementCache('scikit-learn')

def _runif_reasons(*, min_cuda_gpus: int=0, min_torch: Optional[str]=None, max_torch: Optional[str]=None, min_python: Optional[str]=None, bf16_cuda: bool=False, tpu: bool=False, mps: Optional[bool]=None, skip_windows: bool=False, standalone: bool=False, deepspeed: bool=False, dynamo: bool=False, rich: bool=False, omegaconf: bool=False, psutil: bool=False, sklearn: bool=False, onnx: bool=False) -> Tuple[List[str], Dict[str, bool]]:
    if False:
        return 10
    'Construct reasons for pytest skipif.\n\n    Args:\n        min_cuda_gpus: Require this number of gpus and that the ``PL_RUN_CUDA_TESTS=1`` environment variable is set.\n        min_torch: Require that PyTorch is greater or equal than this version.\n        max_torch: Require that PyTorch is less than this version.\n        min_python: Require that Python is greater or equal than this version.\n        bf16_cuda: Require that CUDA device supports bf16.\n        tpu: Require that TPU is available.\n        mps: If True: Require that MPS (Apple Silicon) is available,\n            if False: Explicitly Require that MPS is not available\n        skip_windows: Skip for Windows platform.\n        standalone: Mark the test as standalone, our CI will run it in a separate process.\n            This requires that the ``PL_RUN_STANDALONE_TESTS=1`` environment variable is set.\n        deepspeed: Require that microsoft/DeepSpeed is installed.\n        dynamo: Require that `torch.dynamo` is supported.\n        rich: Require that willmcgugan/rich is installed.\n        omegaconf: Require that omry/omegaconf is installed.\n        psutil: Require that psutil is installed.\n        sklearn: Require that scikit-learn is installed.\n        onnx: Require that onnx is installed.\n\n    '
    (reasons, kwargs) = FabricRunIf(min_cuda_gpus=min_cuda_gpus, min_torch=min_torch, max_torch=max_torch, min_python=min_python, bf16_cuda=bf16_cuda, tpu=tpu, mps=mps, skip_windows=skip_windows, standalone=standalone, deepspeed=deepspeed, dynamo=dynamo)
    if rich and (not _RICH_AVAILABLE):
        reasons.append('Rich')
    if omegaconf and (not _OMEGACONF_AVAILABLE):
        reasons.append('omegaconf')
    if psutil and (not _PSUTIL_AVAILABLE):
        reasons.append('psutil')
    if sklearn and (not _SKLEARN_AVAILABLE):
        reasons.append('scikit-learn')
    if onnx and _TORCH_GREATER_EQUAL_2_0 and (not _ONNX_AVAILABLE):
        reasons.append('onnx')
    return (reasons, kwargs)