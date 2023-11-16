import sys
from .test_common import _maybe_disable_mpi

def test_no_mpi_no_crash():
    if False:
        i = 10
        return i + 15
    with _maybe_disable_mpi(True):
        old_modules = {}
        sb_modules = [name for name in sys.modules.keys() if name.startswith('stable_baselines')]
        for name in sb_modules:
            old_modules[name] = sys.modules.pop(name)
        import stable_baselines
        del stable_baselines
        for (name, mod) in old_modules.items():
            sys.modules[name] = mod