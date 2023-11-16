"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.states.layman as layman
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {layman: {}}

def test_present():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test to verify that the overlay is present.\n    '
    name = 'sunrise'
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    mock = MagicMock(side_effect=[[name], []])
    with patch.dict(layman.__salt__, {'layman.list_local': mock}):
        comt = 'Overlay {} already present'.format(name)
        ret.update({'comment': comt})
        assert layman.present(name) == ret
        with patch.dict(layman.__opts__, {'test': True}):
            comt = 'Overlay {} is set to be added'.format(name)
            ret.update({'comment': comt, 'result': None})
            assert layman.present(name) == ret

def test_absent():
    if False:
        i = 10
        return i + 15
    '\n    Test to verify that the overlay is absent.\n    '
    name = 'sunrise'
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    mock = MagicMock(side_effect=[[], [name]])
    with patch.dict(layman.__salt__, {'layman.list_local': mock}):
        comt = 'Overlay {} already absent'.format(name)
        ret.update({'comment': comt})
        assert layman.absent(name) == ret
        with patch.dict(layman.__opts__, {'test': True}):
            comt = 'Overlay {} is set to be deleted'.format(name)
            ret.update({'comment': comt, 'result': None})
            assert layman.absent(name) == ret