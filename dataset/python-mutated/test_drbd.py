"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.drbd
"""
import pytest
import salt.modules.drbd as drbd
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {drbd: {}}

def test_overview():
    if False:
        print('Hello World!')
    '\n    Test if it shows status of the DRBD devices\n    '
    ret = {'connection state': 'True', 'device': 'Stack', 'fs': 'None', 'local disk state': 'UpToDate', 'local role': 'master', 'minor number': 'Salt', 'mountpoint': 'True', 'partner disk state': 'UpToDate', 'partner role': 'minion', 'percent': '888', 'remains': '666', 'total size': '50', 'used': '50'}
    mock = MagicMock(return_value='Salt:Stack True master/minion UpToDate/UpToDate True None 50 50 666 888')
    with patch.dict(drbd.__salt__, {'cmd.run': mock}):
        assert drbd.overview() == ret
    ret = {'connection state': 'True', 'device': 'Stack', 'local disk state': 'UpToDate', 'local role': 'master', 'minor number': 'Salt', 'partner disk state': 'partner', 'partner role': 'minion', 'synched': '5050', 'synchronisation: ': 'syncbar'}
    mock = MagicMock(return_value='Salt:Stack True master/minion UpToDate/partner syncbar None 50 50')
    with patch.dict(drbd.__salt__, {'cmd.run': mock}):
        assert drbd.overview() == ret

def test_status():
    if False:
        i = 10
        return i + 15
    '\n    Test if it shows status of the DRBD resources via drbdadm\n    '
    ret = [{'local role': 'Primary', 'local volumes': [{'disk': 'UpToDate'}], 'peer nodes': [{'peer volumes': [{'done': '96.47', 'peer-disk': 'Inconsistent', 'replication': 'SyncSource'}], 'peernode name': 'opensuse-node2', 'role': 'Secondary'}], 'resource name': 'single'}]
    mock = MagicMock(return_value='\nsingle role:Primary\n  disk:UpToDate\n  opensuse-node2 role:Secondary\n    replication:SyncSource peer-disk:Inconsistent done:96.47\n')
    with patch.dict(drbd.__salt__, {'cmd.run': mock}):
        assert drbd.status() == ret
    ret = [{'local role': 'Primary', 'local volumes': [{'disk': 'UpToDate', 'volume': '0'}, {'disk': 'UpToDate', 'volume': '1'}], 'peer nodes': [{'peer volumes': [{'peer-disk': 'UpToDate', 'volume': '0'}, {'peer-disk': 'UpToDate', 'volume': '1'}], 'peernode name': 'node2', 'role': 'Secondary'}, {'peer volumes': [{'peer-disk': 'UpToDate', 'volume': '0'}, {'peer-disk': 'UpToDate', 'volume': '1'}], 'peernode name': 'node3', 'role': 'Secondary'}], 'resource name': 'res'}, {'local role': 'Primary', 'local volumes': [{'disk': 'UpToDate', 'volume': '0'}, {'disk': 'UpToDate', 'volume': '1'}], 'peer nodes': [{'peer volumes': [{'peer-disk': 'UpToDate', 'volume': '0'}, {'peer-disk': 'UpToDate', 'volume': '1'}], 'peernode name': 'node2', 'role': 'Secondary'}, {'peer volumes': [{'peer-disk': 'UpToDate', 'volume': '0'}, {'peer-disk': 'UpToDate', 'volume': '1'}], 'peernode name': 'node3', 'role': 'Secondary'}], 'resource name': 'test'}]
    mock = MagicMock(return_value='\nres role:Primary\n  volume:0 disk:UpToDate\n  volume:1 disk:UpToDate\n  node2 role:Secondary\n    volume:0 peer-disk:UpToDate\n    volume:1 peer-disk:UpToDate\n  node3 role:Secondary\n    volume:0 peer-disk:UpToDate\n    volume:1 peer-disk:UpToDate\n\ntest role:Primary\n  volume:0 disk:UpToDate\n  volume:1 disk:UpToDate\n  node2 role:Secondary\n    volume:0 peer-disk:UpToDate\n    volume:1 peer-disk:UpToDate\n  node3 role:Secondary\n    volume:0 peer-disk:UpToDate\n    volume:1 peer-disk:UpToDate\n\n')
    with patch.dict(drbd.__salt__, {'cmd.run': mock}):
        assert drbd.status() == ret