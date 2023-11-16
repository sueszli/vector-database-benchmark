from ...constants import *
from . import cmd, RK_ENCRYPTION

def test_benchmark_crud(archiver, monkeypatch):
    if False:
        print('Hello World!')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    monkeypatch.setenv('_BORG_BENCHMARK_CRUD_TEST', 'YES')
    cmd(archiver, 'benchmark', 'crud', archiver.input_path)