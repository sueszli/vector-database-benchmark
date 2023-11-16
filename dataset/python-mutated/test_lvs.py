"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.lvs
"""
import pytest
import salt.modules.lvs as lvs
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {lvs: {}}

def test_add_service():
    if False:
        while True:
            i = 10
    '\n    Test for Add a virtual service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.add_service() == 'stderr'

def test_edit_service():
    if False:
        print('Hello World!')
    '\n    Test for Edit the virtual service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.edit_service() == 'stderr'

def test_delete_service():
    if False:
        while True:
            i = 10
    '\n    Test for Delete the virtual service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.delete_service() == 'stderr'

def test_add_server():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Add a real server to a virtual service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.add_server() == 'stderr'

def test_edit_server():
    if False:
        i = 10
        return i + 15
    '\n    Test for Edit a real server to a virtual service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.edit_server() == 'stderr'

def test_delete_server():
    if False:
        return 10
    '\n    Test for Delete the realserver from the virtual service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.delete_server() == 'stderr'

def test_clear():
    if False:
        while True:
            i = 10
    '\n    Test for Clear the virtual server table\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
            assert lvs.clear() == 'stderr'

def test_get_rules():
    if False:
        i = 10
        return i + 15
    '\n    Test for Get the virtual server rules\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.dict(lvs.__salt__, {'cmd.run': MagicMock(return_value='A')}):
            assert lvs.get_rules() == 'A'

def test_list_():
    if False:
        print('Hello World!')
    '\n    Test for List the virtual server table\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.list_('p', 's') == 'stderr'

def test_zero():
    if False:
        i = 10
        return i + 15
    '\n    Test for Zero the packet, byte and rate counters in a\n     service or all services.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                assert lvs.zero('p', 's') == 'stderr'

def test_check_service():
    if False:
        return 10
    '\n    Test for Check the virtual service exists.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                with patch.object(lvs, 'get_rules', return_value='C'):
                    assert lvs.check_service('p', 's') == 'Error: service not exists'

def test_check_server():
    if False:
        return 10
    '\n    Test for Check the real server exists in the specified service.\n    '
    with patch.object(lvs, '__detect_os', return_value='C'):
        with patch.object(lvs, '_build_cmd', return_value='B'):
            with patch.dict(lvs.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 'ret', 'stderr': 'stderr'})}):
                with patch.object(lvs, 'get_rules', return_value='C'):
                    assert lvs.check_server('p', 's') == 'Error: server not exists'