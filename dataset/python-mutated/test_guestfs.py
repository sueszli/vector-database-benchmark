"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.guestfs
"""
import pytest
import salt.modules.guestfs as guestfs
from tests.support.mock import MagicMock, call, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {guestfs: {}}

def test_mount():
    if False:
        return 10
    '\n    Test if it mounts an image\n    '
    run_mock = MagicMock(return_value='')
    with patch('os.path.join', MagicMock(return_value='/tmp/guest/fedora.qcow')), patch('os.path.isdir', MagicMock(return_value=False)), patch('os.makedirs', MagicMock()) as makedirs_mock, patch('os.listdir', MagicMock(return_value=False)), patch.dict(guestfs.__salt__, {'cmd.run': run_mock}):
        assert guestfs.mount('/srv/images/fedora.qcow')
        run_mock.assert_called_once_with('guestmount -i -a /srv/images/fedora.qcow --rw /tmp/guest/fedora.qcow', python_shell=False)
        makedirs_mock.assert_called_once()
    run_mock.reset_mock()
    with patch('os.path.join', MagicMock(return_value='/tmp/guest/fedora.qcow')), patch('os.path.isdir', MagicMock(return_value=True)), patch('os.makedirs', MagicMock()) as makedirs_mock, patch('os.listdir', MagicMock(return_value=False)), patch.dict(guestfs.__salt__, {'cmd.run': run_mock}):
        assert guestfs.mount('/srv/images/fedora.qcow')
        run_mock.assert_called_once_with('guestmount -i -a /srv/images/fedora.qcow --rw /tmp/guest/fedora.qcow', python_shell=False)
        makedirs_mock.assert_not_called()
    run_mock.reset_mock()
    with patch('os.path.join', MagicMock(side_effect=['/tmp/guest/fedora.qcow', '/tmp/guest/fedora.qcowabc'])), patch('os.path.isdir', MagicMock(side_effect=[True, False])), patch('os.makedirs', MagicMock()) as makedirs_mock, patch('os.listdir', MagicMock(side_effect=[True, False])), patch.dict(guestfs.__salt__, {'cmd.run': run_mock}):
        assert guestfs.mount('/srv/images/fedora.qcow')
        run_mock.assert_called_once_with('guestmount -i -a /srv/images/fedora.qcow --rw /tmp/guest/fedora.qcowabc', python_shell=False)
        makedirs_mock.assert_called_once()

@pytest.mark.slow_test
def test_umount():
    if False:
        while True:
            i = 10
    '\n    Test the guestfs.unmount function\n    '
    run_mock = MagicMock(side_effect=['', 'lsof output line', ''])
    with patch.dict(guestfs.__salt__, {'cmd.run': run_mock}):
        guestfs.umount('/tmp/mnt/opensuse.qcow', disk='/path/to/opensuse.qcow')
        expected = [call('guestunmount -q /tmp/mnt/opensuse.qcow'), call('lsof /path/to/opensuse.qcow'), call('lsof /path/to/opensuse.qcow')]
        assert expected == run_mock.call_args_list
    run_mock = MagicMock(side_effect=[''])
    with patch.dict(guestfs.__salt__, {'cmd.run': run_mock}):
        guestfs.umount('/tmp/mnt/opensuse.qcow')
        expected = [call('guestunmount -q /tmp/mnt/opensuse.qcow')]
        assert expected == run_mock.call_args_list