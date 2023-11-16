import math
import torch
from torch.utils import benchmark
from torch.utils.benchmark import FuzzedParameter, FuzzedTensor, ParameterAlias
__all__ = ['SpectralOpFuzzer']
MIN_DIM_SIZE = 16
MAX_DIM_SIZE = 16 * 1024

def power_range(upper_bound, base):
    if False:
        return 10
    return (base ** i for i in range(int(math.log(upper_bound, base)) + 1))
REGULAR_SIZES = []
for i in power_range(MAX_DIM_SIZE, 2):
    for j in power_range(MAX_DIM_SIZE // i, 3):
        ij = i * j
        for k in power_range(MAX_DIM_SIZE // ij, 5):
            ijk = ij * k
            if ijk > MIN_DIM_SIZE:
                REGULAR_SIZES.append(ijk)
REGULAR_SIZES.sort()

class SpectralOpFuzzer(benchmark.Fuzzer):

    def __init__(self, *, seed: int, dtype=torch.float64, cuda: bool=False, probability_regular: float=1.0):
        if False:
            print('Hello World!')
        super().__init__(parameters=[FuzzedParameter('ndim', distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True), [FuzzedParameter(name=f'k_any_{i}', minval=MIN_DIM_SIZE, maxval=MAX_DIM_SIZE, distribution='loguniform') for i in range(3)], [FuzzedParameter(name=f'k_regular_{i}', distribution={size: 1.0 / len(REGULAR_SIZES) for size in REGULAR_SIZES}) for i in range(3)], [FuzzedParameter(name=f'k{i}', distribution={ParameterAlias(f'k_regular_{i}'): probability_regular, ParameterAlias(f'k_any_{i}'): 1 - probability_regular}, strict=True) for i in range(3)], [FuzzedParameter(name=f'step_{i}', distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04}) for i in range(3)]], tensors=[FuzzedTensor(name='x', size=('k0', 'k1', 'k2'), steps=('step_0', 'step_1', 'step_2'), probability_contiguous=0.75, min_elements=4 * 1024, max_elements=32 * 1024 ** 2, max_allocation_bytes=2 * 1024 ** 3, dim_parameter='ndim', dtype=dtype, cuda=cuda)], seed=seed)