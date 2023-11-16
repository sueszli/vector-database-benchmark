from __future__ import annotations
import io
import xarray

def test_show_versions() -> None:
    if False:
        return 10
    f = io.StringIO()
    xarray.show_versions(file=f)
    assert 'INSTALLED VERSIONS' in f.getvalue()