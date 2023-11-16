from __future__ import annotations
import pytest
pytest
import bokeh.io.notebook as binb
from tests.support.util.api import verify_all
import bokeh.io as bi
ALL = ('curdoc', 'export_png', 'export_svg', 'export_svgs', 'install_notebook_hook', 'push_notebook', 'output_file', 'output_notebook', 'reset_output', 'save', 'show')
Test___all__ = verify_all(bi, ALL)

def test_jupyter_notebook_hook_installed() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert list(binb._HOOKS) == ['jupyter']
    assert binb._HOOKS['jupyter']['load'] == binb.load_notebook
    assert binb._HOOKS['jupyter']['doc'] == binb.show_doc
    assert binb._HOOKS['jupyter']['app'] == binb.show_app