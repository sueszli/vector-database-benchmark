import importlib
import pytest
from IPython.external.qt_loaders import ID

def test_import_denier():
    if False:
        for i in range(10):
            print('nop')
    ID.forbid('ipython_denied_module')
    with pytest.raises(ImportError, match='disabled by IPython'):
        import ipython_denied_module
    with pytest.raises(ImportError, match='disabled by IPython'):
        importlib.import_module('ipython_denied_module')