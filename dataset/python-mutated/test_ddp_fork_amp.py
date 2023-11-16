import multiprocessing
import torch
from lightning.pytorch.plugins import MixedPrecision
from tests_pytorch.helpers.runif import RunIf

@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True)
def test_amp_gpus_ddp_fork():
    if False:
        print('Hello World!')
    'Ensure the use of AMP with `ddp_fork` (or associated alias strategies) does not generate CUDA initialization\n    errors.'
    _ = MixedPrecision(precision='16-mixed', device='cuda')
    with multiprocessing.get_context('fork').Pool(1) as pool:
        in_bad_fork = pool.apply(torch.cuda._is_in_bad_fork)
    assert not in_bad_fork