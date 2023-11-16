"""
    :codeauthor: :email:`Jorge Schrauwen <sjorge@blackdot.be>`
"""
import textwrap
import salt.grains.smartos as smartos
from tests.support.mock import MagicMock, Mock, mock_open, patch

def test_smartos_computenode_data():
    if False:
        while True:
            i = 10
    '\n    Get a tally of running/stopped zones\n    Output used form a test host with one running\n    and one stopped of each vm type.\n    '
    grains_exp_res = {'computenode_sdc_version': '7.0', 'computenode_vm_capable': True, 'computenode_vm_hw_virt': 'vmx', 'computenode_vms_running': 3, 'computenode_vms_stopped': 3, 'computenode_vms_total': 6, 'computenode_vms_type': {'KVM': 2, 'LX': 2, 'OS': 2}, 'manufacturer': 'Supermicro', 'productname': 'X8STi', 'uuid': '534d4349-0002-2790-2500-2790250054c5'}
    cmd_mock = Mock(side_effect=[textwrap.dedent('            99e40ee7-a8f9-4b57-9225-e7bd19f64b07:test_hvm1:running:BHYV\n            cde351a9-e23d-6856-e268-fff10fe603dc:test_hvm2:stopped:BHYV\n            99e40ee7-a8f9-4b57-9225-e7bd19f64b07:test_hvm3:running:KVM\n            cde351a9-e23d-6856-e268-fff10fe603dc:test_hvm4:stopped:KVM\n            179b50ca-8a4d-4f28-bb08-54b2cd350aa5:test_zone1:running:OS\n            42846fbc-c48a-6390-fd85-d7ac6a76464c:test_zone2:stopped:OS\n            4fd2d7a4-38c4-4068-a2c8-74124364a109:test_zone3:running:LX\n            717abe34-e7b9-4387-820e-0bb041173563:test_zone4:stopped:LX'), textwrap.dedent('            {\n                "Live Image": "20181011T004530Z",\n                "System Type": "SunOS",\n                "Boot Time": "1562528522",\n                "SDC Version": "7.0",\n                "Manufacturer": "Supermicro",\n                "Product": "X8STi",\n                "Serial Number": "1234567890",\n                "SKU Number": "To Be Filled By O.E.M.",\n                "HW Version": "1234567890",\n                "HW Family": "High-End Desktop",\n                "Setup": "false",\n                "VM Capable": true,\n                "Bhyve Capable": false,\n                "Bhyve Max Vcpus": 0,\n                "HVM API": false,\n                "CPU Type": "Intel(R) Xeon(R) CPU W3520 @ 2.67GHz",\n                "CPU Virtualization": "vmx",\n                "CPU Physical Cores": 1,\n                "Admin NIC Tag": "",\n                "UUID": "534d4349-0002-2790-2500-2790250054c5",\n                "Hostname": "sdc",\n                "CPU Total Cores": 8,\n                "MiB of Memory": "16375",\n                "Zpool": "zones",\n                "Zpool Disks": "c1t0d0,c1t1d0",\n                "Zpool Profile": "mirror",\n                "Zpool Creation": 1406392163,\n                "Zpool Size in GiB": 1797,\n                "Disks": {\n                "c1t0d0": {"Size in GB": 2000},\n                "c1t1d0": {"Size in GB": 2000}\n                },\n                "Boot Parameters": {\n                "smartos": "true",\n                "console": "text",\n                "boot_args": "",\n                "bootargs": ""\n                },\n                "Network Interfaces": {\n                "e1000g0": {"MAC Address": "00:00:00:00:00:01", "ip4addr": "123.123.123.123", "Link Status": "up", "NIC Names": ["admin"]},\n                "e1000g1": {"MAC Address": "00:00:00:00:00:05", "ip4addr": "", "Link Status": "down", "NIC Names": []}\n                },\n                "Virtual Network Interfaces": {\n                },\n                "Link Aggregations": {\n                }\n            }')])
    with patch.dict(smartos.__salt__, {'cmd.run': cmd_mock}):
        grains_res = smartos._smartos_computenode_data()
        assert grains_exp_res == grains_res

def test_smartos_zone_data():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get basic information about a non-global zone\n    '
    grains_exp_res = {'imageversion': 'pkgbuild 18.1.0', 'zoneid': '5', 'zonename': 'dda70f61-70fe-65e7-cf70-d878d69442d4'}
    cmd_mock = Mock(side_effect=['5:dda70f61-70fe-65e7-cf70-d878d69442d4:running:/:dda70f61-70fe-65e7-cf70-d878d69442d4:native:excl:0'])
    fopen_mock = mock_open(read_data={'/etc/product': textwrap.dedent('            Name: Joyent Instance\n            Image: pkgbuild 18.1.0\n            Documentation: https://docs.joyent.com/images/smartos/pkgbuild\n            ')})
    with patch.dict(smartos.__salt__, {'cmd.run': cmd_mock}), patch('os.path.isfile', MagicMock(return_value=True)), patch('salt.utils.files.fopen', fopen_mock):
        grains_res = smartos._smartos_zone_data()
        assert grains_exp_res == grains_res

def test_smartos_zone_pkgsrc_data_in_zone():
    if False:
        while True:
            i = 10
    '\n    Get pkgsrc information from a zone\n    '
    grains_exp_res = {'pkgsrcpath': 'https://pkgsrc.joyent.com/packages/SmartOS/2018Q1/x86_64/All', 'pkgsrcversion': '2018Q1'}
    isfile_mock = Mock(side_effect=[True, False])
    fopen_mock = mock_open(read_data={'/opt/local/etc/pkg_install.conf': textwrap.dedent('            GPG_KEYRING_VERIFY=/opt/local/etc/gnupg/pkgsrc.gpg\n            GPG_KEYRING_PKGVULN=/opt/local/share/gnupg/pkgsrc-security.gpg\n            PKG_PATH=https://pkgsrc.joyent.com/packages/SmartOS/2018Q1/x86_64/All\n            ')})
    with patch('os.path.isfile', isfile_mock), patch('salt.utils.files.fopen', fopen_mock):
        grains_res = smartos._smartos_zone_pkgsrc_data()
        assert grains_exp_res == grains_res

def test_smartos_zone_pkgsrc_data_in_globalzone():
    if False:
        i = 10
        return i + 15
    '\n    Get pkgsrc information from the globalzone\n    '
    grains_exp_res = {'pkgsrcpath': 'https://pkgsrc.joyent.com/packages/SmartOS/trunk/tools/All', 'pkgsrcversion': 'trunk'}
    isfile_mock = Mock(side_effect=[False, True])
    fopen_mock = mock_open(read_data={'/opt/tools/etc/pkg_install.conf': textwrap.dedent('            GPG_KEYRING_PKGVULN=/opt/tools/share/gnupg/pkgsrc-security.gpg\n            GPG_KEYRING_VERIFY=/opt/tools/etc/gnupg/pkgsrc.gpg\n            PKG_PATH=https://pkgsrc.joyent.com/packages/SmartOS/trunk/tools/All\n            VERIFIED_INSTALLATION=always\n            ')})
    with patch('os.path.isfile', isfile_mock), patch('salt.utils.files.fopen', fopen_mock):
        grains_res = smartos._smartos_zone_pkgsrc_data()
        assert grains_exp_res == grains_res

def test_smartos_zone_pkgin_data_in_zone():
    if False:
        print('Hello World!')
    '\n    Get pkgin information from a zone\n    '
    grains_exp_res = {'pkgin_repositories': ['https://pkgsrc.joyent.com/packages/SmartOS/2018Q1/x86_64/All', 'http://pkg.blackdot.be/packages/2018Q1/x86_64/All']}
    isfile_mock = Mock(side_effect=[True, False])
    fopen_mock = mock_open(read_data={'/opt/local/etc/pkgin/repositories.conf': textwrap.dedent('            # $Id: repositories.conf,v 1.3 2012/06/13 13:50:17 imilh Exp $\n            #\n            # Pkgin repositories list\n            #\n            # Simply add repositories URIs one below the other\n            #\n            # WARNING: order matters, duplicates will not be added, if two\n            # repositories hold the same package, it will be fetched from\n            # the first one listed in this file.\n            #\n            # This file format supports the following macros:\n            # $arch to define the machine hardware platform\n            # $osrelease to define the release version for the operating system\n            #\n            # Remote ftp repository\n            #\n            # ftp://ftp.netbsd.org/pub/pkgsrc/packages/NetBSD/$arch/5.1/All\n            #\n            # Remote http repository\n            #\n            # http://mirror-master.dragonflybsd.org/packages/$arch/DragonFly-$osrelease/stable/All\n            #\n            # Local repository (must contain a pkg_summary.gz or bz2)\n            #\n            # file:///usr/pkgsrc/packages/All\n            #\n            https://pkgsrc.joyent.com/packages/SmartOS/2018Q1/x86_64/All\n            http://pkg.blackdot.be/packages/2018Q1/x86_64/All\n            ')})
    with patch('os.path.isfile', isfile_mock), patch('salt.utils.files.fopen', fopen_mock):
        grains_res = smartos._smartos_zone_pkgin_data()
        assert grains_exp_res == grains_res

def test_smartos_zone_pkgin_data_in_globalzone():
    if False:
        i = 10
        return i + 15
    '\n    Get pkgin information from the globalzone\n    '
    grains_exp_res = {'pkgin_repositories': ['https://pkgsrc.joyent.com/packages/SmartOS/trunk/tools/All']}
    isfile_mock = Mock(side_effect=[False, True])
    fopen_mock = mock_open(read_data={'/opt/tools/etc/pkgin/repositories.conf': textwrap.dedent('            #\n            # Pkgin repositories list\n            #\n            # Simply add repositories URIs one below the other\n            #\n            # WARNING: order matters, duplicates will not be added, if two\n            # repositories hold the same package, it will be fetched from\n            # the first one listed in this file.\n            #\n            # This file format supports the following macros:\n            # $arch to define the machine hardware platform\n            # $osrelease to define the release version for the operating system\n            #\n            # Remote ftp repository\n            #\n            # ftp://ftp.netbsd.org/pub/pkgsrc/packages/NetBSD/$arch/5.1/All\n            #\n            # Remote http repository\n            #\n            # http://mirror-master.dragonflybsd.org/packages/$arch/DragonFly-$osrelease/stable/All\n            #\n            # Local repository (must contain a pkg_summary.gz or bz2)\n            #\n            # file:///usr/pkgsrc/packages/All\n            #\n            https://pkgsrc.joyent.com/packages/SmartOS/trunk/tools/All\n            ')})
    with patch('os.path.isfile', isfile_mock), patch('salt.utils.files.fopen', fopen_mock):
        grains_res = smartos._smartos_zone_pkgin_data()
        assert grains_exp_res == grains_res