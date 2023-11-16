"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.states.boto_lc as boto_lc
from salt.exceptions import SaltInvocationError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {boto_lc: {}}

def test_present():
    if False:
        i = 10
        return i + 15
    '\n    Test to ensure the launch configuration exists.\n    '
    name = 'mylc'
    image_id = 'ami-0b9c9f62'
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': ''}
    pytest.raises(SaltInvocationError, boto_lc.present, name, image_id, user_data=True, cloud_init=True)
    mock = MagicMock(side_effect=[True, False])
    with patch.dict(boto_lc.__salt__, {'boto_asg.launch_configuration_exists': mock}):
        comt = 'Launch configuration present.'
        ret.update({'comment': comt})
        assert boto_lc.present(name, image_id) == ret
        with patch.dict(boto_lc.__opts__, {'test': True}):
            comt = 'Launch configuration set to be created.'
            ret.update({'comment': comt, 'result': None})
            assert boto_lc.present(name, image_id) == ret

def test_absent():
    if False:
        i = 10
        return i + 15
    '\n    Test to ensure the named launch configuration is deleted.\n    '
    name = 'mylc'
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': ''}
    mock = MagicMock(side_effect=[False, True])
    with patch.dict(boto_lc.__salt__, {'boto_asg.launch_configuration_exists': mock}):
        comt = 'Launch configuration does not exist.'
        ret.update({'comment': comt})
        assert boto_lc.absent(name) == ret
        with patch.dict(boto_lc.__opts__, {'test': True}):
            comt = 'Launch configuration set to be deleted.'
            ret.update({'comment': comt, 'result': None})
            assert boto_lc.absent(name) == ret