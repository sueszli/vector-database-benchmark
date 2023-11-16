from __future__ import annotations
import pytest
pytest
from unittest.mock import call, patch
import bokeh.util.sampledata as bus

@pytest.mark.sampledata
def test_external_path_bad() -> None:
    if False:
        while True:
            i = 10
    pat = 'Could not locate external data file (.*)junkjunk. Please execute bokeh.sampledata.download()'
    with pytest.raises(RuntimeError, match=pat):
        bus.external_path('junkjunk')

@pytest.mark.sampledata
def test_package_dir() -> None:
    if False:
        i = 10
        return i + 15
    path = bus.package_dir()
    assert path.exists()
    assert path.parts[-2:] == ('sampledata', '_data')

@pytest.mark.sampledata
def test_package_csv() -> None:
    if False:
        print('Hello World!')
    with patch('pandas.read_csv') as mock_read_csv:
        bus.package_csv('module', 'foo', bar=10)
    assert mock_read_csv.has_call(call(bus.package_path('foo'), bar=10))

@pytest.mark.sampledata
def test_package_path() -> None:
    if False:
        return 10
    assert bus.package_path('foo') == bus.package_dir() / 'foo'

@pytest.mark.sampledata
def test_open_csv() -> None:
    if False:
        for i in range(10):
            print('nop')
    with patch('builtins.open') as mock_open:
        bus.open_csv('foo')
    assert mock_open.has_call(call('foo', 'r', newline='', encoding='utf8'))