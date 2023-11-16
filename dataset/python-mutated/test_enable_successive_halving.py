"""Tests for making sure experimental imports work as expected."""
import textwrap
import pytest
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import assert_run_python_script

@pytest.mark.xfail(_IS_WASM, reason='cannot start subprocess')
def test_imports_strategies():
    if False:
        print('Hello World!')
    good_import = '\n    from sklearn.experimental import enable_halving_search_cv\n    from sklearn.model_selection import HalvingGridSearchCV\n    from sklearn.model_selection import HalvingRandomSearchCV\n    '
    assert_run_python_script(textwrap.dedent(good_import))
    good_import_with_model_selection_first = '\n    import sklearn.model_selection\n    from sklearn.experimental import enable_halving_search_cv\n    from sklearn.model_selection import HalvingGridSearchCV\n    from sklearn.model_selection import HalvingRandomSearchCV\n    '
    assert_run_python_script(textwrap.dedent(good_import_with_model_selection_first))
    bad_imports = "\n    import pytest\n\n    with pytest.raises(ImportError, match='HalvingGridSearchCV is experimental'):\n        from sklearn.model_selection import HalvingGridSearchCV\n\n    import sklearn.experimental\n    with pytest.raises(ImportError, match='HalvingRandomSearchCV is experimental'):\n        from sklearn.model_selection import HalvingRandomSearchCV\n    "
    assert_run_python_script(textwrap.dedent(bad_imports))