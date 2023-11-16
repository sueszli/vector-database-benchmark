import sys
import subprocess
import textwrap
from importlib import reload
import pickle
import pytest
import numpy.exceptions as ex
from numpy.testing import assert_raises, assert_warns, assert_, assert_equal, IS_WASM

def test_numpy_reloading():
    if False:
        while True:
            i = 10
    import numpy as np
    import numpy._globals
    _NoValue = np._NoValue
    VisibleDeprecationWarning = ex.VisibleDeprecationWarning
    ModuleDeprecationWarning = ex.ModuleDeprecationWarning
    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is ex.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is ex.VisibleDeprecationWarning)
    assert_raises(RuntimeError, reload, numpy._globals)
    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is ex.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is ex.VisibleDeprecationWarning)

def test_novalue():
    if False:
        while True:
            i = 10
    import numpy as np
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        assert_equal(repr(np._NoValue), '<no value>')
        assert_(pickle.loads(pickle.dumps(np._NoValue, protocol=proto)) is np._NoValue)

@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
def test_full_reimport():
    if False:
        while True:
            i = 10
    'At the time of writing this, it is *not* truly supported, but\n    apparently enough users rely on it, for it to be an annoying change\n    when it started failing previously.\n    '
    code = textwrap.dedent('\n        import sys\n        from pytest import warns\n        import numpy as np\n\n        for k in list(sys.modules.keys()):\n            if "numpy" in k:\n                del sys.modules[k]\n\n        with warns(UserWarning):\n            import numpy as np\n        ')
    p = subprocess.run([sys.executable, '-c', code], capture_output=True)
    if p.returncode:
        raise AssertionError(f'Non-zero return code: {p.returncode!r}\n\n{p.stderr.decode()}')