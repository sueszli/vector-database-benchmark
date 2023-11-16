"""Tests for making sure experimental imports work as expected."""
import textwrap
import pytest
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import assert_run_python_script

@pytest.mark.xfail(_IS_WASM, reason='cannot start subprocess')
def test_import_raises_warning():
    if False:
        while True:
            i = 10
    code = '\n    import pytest\n    with pytest.warns(UserWarning, match="it is not needed to import"):\n        from sklearn.experimental import enable_hist_gradient_boosting  # noqa\n    '
    assert_run_python_script(textwrap.dedent(code))