import pytest
from lightning.fabric.utilities.testing import _runif_reasons

def RunIf(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    (reasons, marker_kwargs) = _runif_reasons(**kwargs)
    return pytest.mark.skipif(condition=len(reasons) > 0, reason=f"Requires: [{' + '.join(reasons)}]", **marker_kwargs)