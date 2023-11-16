import pytest
from vyper.codegen.context import Context
from vyper.codegen.core import get_type_for_exact_size

class ContextMock(Context):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._mock_vars = False
        self._size = 0

    def internal_memory_scope(self):
        if False:
            while True:
                i = 10
        if not self._mock_vars:
            for i in range(20):
                self._new_variable(f'#mock{i}', get_type_for_exact_size(self._size), self._size, bool(i % 2))
            self._mock_vars = True
        return super().internal_memory_scope()

    @classmethod
    def set_mock_var_size(cls, size):
        if False:
            while True:
                i = 10
        cls._size = size * 32

def pytest_addoption(parser):
    if False:
        i = 10
        return i + 15
    parser.addoption('--memorymock', action='store_true', help='Run tests with mock allocated vars')

def pytest_generate_tests(metafunc):
    if False:
        for i in range(10):
            print('nop')
    if 'memory_mocker' in metafunc.fixturenames:
        params = range(1, 11, 2) if metafunc.config.getoption('memorymock') else [False]
        metafunc.parametrize('memory_mocker', params, indirect=True)

def pytest_collection_modifyitems(items, config):
    if False:
        i = 10
        return i + 15
    if config.getoption('memorymock'):
        for item in list(items):
            if 'memory_mocker' not in item.fixturenames:
                items.remove(item)
        config.pluginmanager.get_plugin('terminalreporter')._numcollected = len(items)

@pytest.fixture
def memory_mocker(monkeypatch, request):
    if False:
        return 10
    if request.param:
        monkeypatch.setattr('vyper.codegen.context.Context', ContextMock)
        ContextMock.set_mock_var_size(request.param)