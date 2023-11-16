"""
    :codeauthor: :email:`Shane Lee <slee@saltstack.com>`
"""
import errno
import textwrap
import salt.grains.iscsi as iscsi
from tests.support.mock import MagicMock, mock_open, patch

def test_windows_iscsi_iqn_grains():
    if False:
        print('Hello World!')
    cmd_run_mock = MagicMock(return_value={'stdout': 'iSCSINodeName\niqn.1991-05.com.microsoft:simon-x1\n'})
    _grains = {}
    with patch('salt.utils.path.which', MagicMock(return_value=True)):
        with patch('salt.modules.cmdmod.run_all', cmd_run_mock):
            _grains['iscsi_iqn'] = iscsi._windows_iqn()
    assert _grains.get('iscsi_iqn') == ['iqn.1991-05.com.microsoft:simon-x1']

def test_aix_iscsi_iqn_grains():
    if False:
        while True:
            i = 10
    cmd_run_mock = MagicMock(return_value='initiator_name iqn.localhost.hostid.7f000001')
    _grains = {}
    with patch('salt.modules.cmdmod.run', cmd_run_mock):
        _grains['iscsi_iqn'] = iscsi._aix_iqn()
    assert _grains.get('iscsi_iqn') == ['iqn.localhost.hostid.7f000001']

def test_linux_iscsi_iqn_grains():
    if False:
        return 10
    _iscsi_file = textwrap.dedent('        ## DO NOT EDIT OR REMOVE THIS FILE!\n        ## If you remove this file, the iSCSI daemon will not start.\n        ## If you change the InitiatorName, existing access control lists\n        ## may reject this initiator.  The InitiatorName must be unique\n        ## for each iSCSI initiator.  Do NOT duplicate iSCSI InitiatorNames.\n        InitiatorName=iqn.1993-08.org.debian:01:d12f7aba36\n        ')
    with patch('salt.utils.files.fopen', mock_open(read_data=_iscsi_file)):
        iqn = iscsi._linux_iqn()
    assert isinstance(iqn, list)
    assert len(iqn) == 1
    assert iqn == ['iqn.1993-08.org.debian:01:d12f7aba36']

def test_linux_iqn_non_root():
    if False:
        print('Hello World!')
    '\n    Test if linux_iqn is running on salt-master as non-root\n    and handling access denial properly.\n    :return:\n    '
    with patch('salt.utils.files.fopen', side_effect=IOError(errno.EPERM, 'The cables are not the same length.')):
        with patch('salt.grains.iscsi.log'):
            assert iscsi._linux_iqn() == []
            iscsi.log.debug.assert_called()
            assert 'Error while accessing' in iscsi.log.debug.call_args[0][0]
            assert 'cables are not the same' in iscsi.log.debug.call_args[0][2].strerror
            assert iscsi.log.debug.call_args[0][2].errno == errno.EPERM
            assert iscsi.log.debug.call_args[0][1] == '/etc/iscsi/initiatorname.iscsi'

def test_linux_iqn_no_iscsii_initiator():
    if False:
        i = 10
        return i + 15
    '\n    Test if linux_iqn is running on salt-master as root.\n    iscsii initiator is not there accessible or is not supported.\n    :return:\n    '
    with patch('salt.utils.files.fopen', side_effect=IOError(errno.ENOENT, '')):
        with patch('salt.grains.iscsi.log'):
            assert iscsi._linux_iqn() == []
            iscsi.log.debug.assert_not_called()