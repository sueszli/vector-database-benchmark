"""Tests for making sure experimental imports work as expected."""
import textwrap
import pytest
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import assert_run_python_script

@pytest.mark.xfail(_IS_WASM, reason='cannot start subprocess')
def test_imports_strategies():
    if False:
        return 10
    good_import = '\n    from sklearn.experimental import enable_iterative_imputer\n    from sklearn.impute import IterativeImputer\n    '
    assert_run_python_script(textwrap.dedent(good_import))
    good_import_with_ensemble_first = '\n    import sklearn.ensemble\n    from sklearn.experimental import enable_iterative_imputer\n    from sklearn.impute import IterativeImputer\n    '
    assert_run_python_script(textwrap.dedent(good_import_with_ensemble_first))
    bad_imports = "\n    import pytest\n\n    with pytest.raises(ImportError, match='IterativeImputer is experimental'):\n        from sklearn.impute import IterativeImputer\n\n    import sklearn.experimental\n    with pytest.raises(ImportError, match='IterativeImputer is experimental'):\n        from sklearn.impute import IterativeImputer\n    "
    assert_run_python_script(textwrap.dedent(bad_imports))