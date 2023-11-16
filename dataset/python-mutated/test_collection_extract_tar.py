from __future__ import annotations
import pytest
from ansible.errors import AnsibleError
from ansible.galaxy.collection import _extract_tar_dir

@pytest.fixture
def fake_tar_obj(mocker):
    if False:
        return 10
    m_tarfile = mocker.Mock()
    m_tarfile._ansible_normalized_cache = {'/some/dir': mocker.Mock()}
    m_tarfile.type = mocker.Mock(return_value=b'99')
    m_tarfile.SYMTYPE = mocker.Mock(return_value=b'22')
    return m_tarfile

def test_extract_tar_member_trailing_sep(mocker):
    if False:
        for i in range(10):
            print('nop')
    m_tarfile = mocker.Mock()
    m_tarfile._ansible_normalized_cache = {}
    with pytest.raises(AnsibleError, match='Unable to extract'):
        _extract_tar_dir(m_tarfile, '/some/dir/', b'/some/dest')

def test_extract_tar_dir_exists(mocker, fake_tar_obj):
    if False:
        print('Hello World!')
    mocker.patch('os.makedirs', return_value=None)
    m_makedir = mocker.patch('os.mkdir', return_value=None)
    mocker.patch('os.path.isdir', return_value=True)
    _extract_tar_dir(fake_tar_obj, '/some/dir', b'/some/dest')
    assert not m_makedir.called

def test_extract_tar_dir_does_not_exist(mocker, fake_tar_obj):
    if False:
        while True:
            i = 10
    mocker.patch('os.makedirs', return_value=None)
    m_makedir = mocker.patch('os.mkdir', return_value=None)
    mocker.patch('os.path.isdir', return_value=False)
    _extract_tar_dir(fake_tar_obj, '/some/dir', b'/some/dest')
    assert m_makedir.called
    assert m_makedir.call_args[0] == (b'/some/dir', 493)