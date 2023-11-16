from pathlib import Path
from textual.file_monitor import FileMonitor

def test_repr() -> None:
    if False:
        for i in range(10):
            print('nop')
    file_monitor = FileMonitor([Path('.')], lambda : None)
    assert 'FileMonitor' in repr(file_monitor)