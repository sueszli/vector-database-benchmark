import io
import sys
import pytest
from rich.console import Console
from rich.file_proxy import FileProxy

def test_empty_bytes():
    if False:
        print('Hello World!')
    console = Console()
    file_proxy = FileProxy(console, sys.stdout)
    with pytest.raises(TypeError):
        file_proxy.write(b'')
    with pytest.raises(TypeError):
        file_proxy.write(b'foo')

def test_flush():
    if False:
        while True:
            i = 10
    file = io.StringIO()
    console = Console(file=file)
    file_proxy = FileProxy(console, file)
    file_proxy.write('foo')
    assert file.getvalue() == ''
    file_proxy.flush()
    assert file.getvalue() == 'foo\n'

def test_new_lines():
    if False:
        while True:
            i = 10
    file = io.StringIO()
    console = Console(file=file)
    file_proxy = FileProxy(console, file)
    file_proxy.write('-\n-')
    assert file.getvalue() == '-\n'
    file_proxy.flush()
    assert file.getvalue() == '-\n-\n'