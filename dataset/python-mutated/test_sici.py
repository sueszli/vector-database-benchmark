import numpy as np
import scipy.special as sc
from scipy.special._testutils import FuncData

def test_sici_consistency():
    if False:
        for i in range(10):
            print('nop')

    def sici(x):
        if False:
            i = 10
            return i + 15
        (si, ci) = sc.sici(x + 0j)
        return (si.real, ci.real)
    x = np.r_[-np.logspace(8, -30, 200), 0, np.logspace(-30, 8, 200)]
    (si, ci) = sc.sici(x)
    dataset = np.column_stack((x, si, ci))
    FuncData(sici, dataset, 0, (1, 2), rtol=1e-12).check()

def test_shichi_consistency():
    if False:
        print('Hello World!')

    def shichi(x):
        if False:
            while True:
                i = 10
        (shi, chi) = sc.shichi(x + 0j)
        return (shi.real, chi.real)
    x = np.r_[-np.logspace(np.log10(700), -30, 200), 0, np.logspace(-30, np.log10(700), 200)]
    (shi, chi) = sc.shichi(x)
    dataset = np.column_stack((x, shi, chi))
    FuncData(shichi, dataset, 0, (1, 2), rtol=1e-14).check()