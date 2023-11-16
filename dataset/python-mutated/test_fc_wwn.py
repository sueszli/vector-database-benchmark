from __future__ import annotations
from ansible.module_utils.facts.network import fc_wwn
from unittest.mock import Mock
LSDEV_OUTPUT = '\nfcs0 Defined   00-00 8Gb PCI Express Dual Port FC Adapter (df1000f114108a03)\nfcs1 Available 04-00 8Gb PCI Express Dual Port FC Adapter (df1000f114108a03)\n'
LSCFG_OUTPUT = '\n  fcs1             U78CB.001.WZS00ZS-P1-C9-T1  8Gb PCI Express Dual Port FC Adapter (df1000f114108a03)\n\n        Part Number.................00E0806\n        Serial Number...............1C4090830F\n        Manufacturer................001C\n        EC Level.................... D77161\n        Customer Card ID Number.....577D\n        FRU Number..................00E0806\n        Device Specific.(ZM)........3\n        Network Address.............10000090FA551508\n        ROS Level and ID............027820B7\n        Device Specific.(Z0)........31004549\n        Device Specific.(ZC)........00000000\n        Hardware Location Code......U78CB.001.WZS00ZS-P1-C9-T1\n'
FCINFO_OUTPUT = '\nHBA Port WWN: 10000090fa1658de\n        Port Mode: Initiator\n        Port ID: 30100\n        OS Device Name: /dev/cfg/c13\n        Manufacturer: Emulex\n        Model: LPe12002-S\n        Firmware Version: LPe12002-S 2.01a12\n        FCode/BIOS Version: Boot:5.03a0 Fcode:3.01a1\n        Serial Number: 4925381+13090001ER\n        Driver Name: emlxs\n        Driver Version: 3.3.00.1 (2018.01.05.16.30)\n        Type: N-port\n        State: online\n        Supported Speeds: 2Gb 4Gb 8Gb\n        Current Speed: 8Gb\n        Node WWN: 20000090fa1658de\n        NPIV Not Supported\n'
IOSCAN_OUT = '\nClass     I  H/W Path    Driver S/W State   H/W Type     Description\n==================================================================\nfc        0  2/0/10/1/0  fcd   CLAIMED     INTERFACE    HP AB379-60101 4Gb Dual Port PCI/PCI-X Fibre Channel Adapter (FC Port 1)\n                        /dev/fcd0\n'
FCMSUTIL_OUT = '\n                           Vendor ID is = 0x1077\n                           Device ID is = 0x2422\n            PCI Sub-system Vendor ID is = 0x103C\n                   PCI Sub-system ID is = 0x12D7\n                               PCI Mode = PCI-X 133 MHz\n                       ISP Code version = 5.4.0\n                       ISP Chip version = 3\n                               Topology = PTTOPT_FABRIC\n                             Link Speed = 4Gb\n                     Local N_Port_id is = 0x010300\n                  Previous N_Port_id is = None\n            N_Port Node World Wide Name = 0x50060b00006975ed\n            N_Port Port World Wide Name = 0x50060b00006975ec\n            Switch Port World Wide Name = 0x200300051e046c0f\n            Switch Node World Wide Name = 0x100000051e046c0f\n              N_Port Symbolic Port Name = server1_fcd0\n              N_Port Symbolic Node Name = server1_HP-UX_B.11.31\n                           Driver state = ONLINE\n                       Hardware Path is = 2/0/10/1/0\n                     Maximum Frame Size = 2048\n         Driver-Firmware Dump Available = NO\n         Driver-Firmware Dump Timestamp = N/A\n                                   TYPE = PFC\n                         NPIV Supported = YES\n                         Driver Version = @(#) fcd B.11.31.1103 Dec  6 2010\n'

def mock_get_bin_path(cmd, required=False, opt_dirs=None):
    if False:
        i = 10
        return i + 15
    result = None
    if cmd == 'lsdev':
        result = '/usr/sbin/lsdev'
    elif cmd == 'lscfg':
        result = '/usr/sbin/lscfg'
    elif cmd == 'fcinfo':
        result = '/usr/sbin/fcinfo'
    elif cmd == 'ioscan':
        result = '/usr/bin/ioscan'
    elif cmd == 'fcmsutil':
        result = '/opt/fcms/bin/fcmsutil'
    return result

def mock_run_command(cmd):
    if False:
        for i in range(10):
            print('nop')
    rc = 0
    if 'lsdev' in cmd:
        result = LSDEV_OUTPUT
    elif 'lscfg' in cmd:
        result = LSCFG_OUTPUT
    elif 'fcinfo' in cmd:
        result = FCINFO_OUTPUT
    elif 'ioscan' in cmd:
        result = IOSCAN_OUT
    elif 'fcmsutil' in cmd:
        result = FCMSUTIL_OUT
    else:
        rc = 1
        result = 'Error'
    return (rc, result, '')

def test_get_fc_wwn_info(mocker):
    if False:
        print('Hello World!')
    module = Mock()
    inst = fc_wwn.FcWwnInitiatorFactCollector()
    mocker.patch.object(module, 'get_bin_path', side_effect=mock_get_bin_path)
    mocker.patch.object(module, 'run_command', side_effect=mock_run_command)
    d = {'aix6': ['10000090FA551508'], 'sunos5': ['10000090fa1658de'], 'hp-ux11': ['0x50060b00006975ec']}
    for (key, value) in d.items():
        mocker.patch('sys.platform', key)
        wwn_expected = {'fibre_channel_wwn': value}
        assert wwn_expected == inst.collect(module=module)