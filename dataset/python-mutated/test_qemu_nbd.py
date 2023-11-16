"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import glob
import os.path
import tempfile
import pytest
import salt.modules.qemu_nbd as qemu_nbd
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {qemu_nbd: {}}

def test_connect():
    if False:
        while True:
            i = 10
    '\n    Test if it activate nbd for an image file.\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(qemu_nbd.__salt__, {'cmd.run': mock}):
        with patch.object(os.path, 'isfile', MagicMock(return_value=False)):
            assert qemu_nbd.connect('/tmp/image.raw') == ''
            assert qemu_nbd.connect('/tmp/image.raw') == ''
    with patch.object(os.path, 'isfile', mock):
        with patch.object(glob, 'glob', MagicMock(return_value=['/dev/nbd0'])):
            with patch.dict(qemu_nbd.__salt__, {'cmd.run': mock, 'cmd.retcode': MagicMock(side_effect=[1, 0])}):
                assert qemu_nbd.connect('/tmp/image.raw') == '/dev/nbd0'
            with patch.dict(qemu_nbd.__salt__, {'cmd.run': mock, 'cmd.retcode': MagicMock(return_value=False)}):
                assert qemu_nbd.connect('/tmp/image.raw') == ''

def test_mount():
    if False:
        while True:
            i = 10
    '\n    Test if it pass in the nbd connection device location,\n    mount all partitions and return a dict of mount points.\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(qemu_nbd.__salt__, {'cmd.run': mock}):
        assert qemu_nbd.mount('/dev/nbd0') == {}

@pytest.mark.slow_test
def test_init():
    if False:
        return 10
    '\n    Test if it mount the named image via qemu-nbd\n    and return the mounted roots\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(qemu_nbd.__salt__, {'cmd.run': mock}):
        assert qemu_nbd.init('/srv/image.qcow2') == ''
    with patch.object(os.path, 'isfile', mock), patch.object(glob, 'glob', MagicMock(return_value=['/dev/nbd0'])), patch.dict(qemu_nbd.__salt__, {'cmd.run': mock, 'mount.mount': mock, 'cmd.retcode': MagicMock(side_effect=[1, 0])}):
        expected = {os.sep.join([tempfile.gettempdir(), 'nbd', 'nbd0', 'nbd0']): '/dev/nbd0'}
        assert qemu_nbd.init('/srv/image.qcow2') == expected

def test_clear():
    if False:
        return 10
    '\n    Test if it pass in the mnt dict returned from nbd_mount\n    to unmount and disconnect the image from nbd.\n    '
    mock_run = MagicMock(return_value=True)
    with patch.dict(qemu_nbd.__salt__, {'cmd.run': mock_run, 'mount.umount': MagicMock(side_effect=[False, True])}):
        assert qemu_nbd.clear({'/mnt/foo': '/dev/nbd0p1'}) == {'/mnt/foo': '/dev/nbd0p1'}
        assert qemu_nbd.clear({'/mnt/foo': '/dev/nbd0p1'}) == {}