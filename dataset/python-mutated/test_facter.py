from __future__ import annotations
from unittest.mock import Mock, patch
from ..base import BaseFactsTest
from ansible.module_utils.facts.other.facter import FacterFactCollector
facter_json_output = '\n{\n  "operatingsystemmajrelease": "25",\n  "hardwareisa": "x86_64",\n  "kernel": "Linux",\n  "path": "/home/testuser/src/ansible/bin:/home/testuser/perl5/bin:/home/testuser/perl5/bin:/home/testuser/bin:/home/testuser/.local/bin:/home/testuser/pythons/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/testuser/.cabal/bin:/home/testuser/gopath/bin:/home/testuser/.rvm/bin",\n  "memorysize": "15.36 GB",\n  "memoryfree": "4.88 GB",\n  "swapsize": "7.70 GB",\n  "swapfree": "6.75 GB",\n  "swapsize_mb": "7880.00",\n  "swapfree_mb": "6911.41",\n  "memorysize_mb": "15732.95",\n  "memoryfree_mb": "4997.68",\n  "lsbmajdistrelease": "25",\n  "macaddress": "02:42:ea:15:d8:84",\n  "id": "testuser",\n  "domain": "example.com",\n  "augeasversion": "1.7.0",\n  "os": {\n    "name": "Fedora",\n    "family": "RedHat",\n    "release": {\n      "major": "25",\n      "full": "25"\n    },\n    "lsb": {\n      "distcodename": "TwentyFive",\n      "distid": "Fedora",\n      "distdescription": "Fedora release 25 (Twenty Five)",\n      "release": ":core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch",\n      "distrelease": "25",\n      "majdistrelease": "25"\n    }\n  },\n  "processors": {\n    "models": [\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n      "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz"\n    ],\n    "count": 8,\n    "physicalcount": 1\n  },\n  "architecture": "x86_64",\n  "hardwaremodel": "x86_64",\n  "operatingsystem": "Fedora",\n  "processor0": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor1": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor2": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor3": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor4": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor5": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor6": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processor7": "Intel(R) Core(TM) i7-4800MQ CPU @ 2.70GHz",\n  "processorcount": 8,\n  "uptime_seconds": 1558090,\n  "fqdn": "myhostname.example.com",\n  "rubyversion": "2.3.3",\n  "gid": "testuser",\n  "physicalprocessorcount": 1,\n  "netmask": "255.255.0.0",\n  "uniqueid": "a8c01301",\n  "uptime_days": 18,\n  "interfaces": "docker0,em1,lo,vethf20ff12,virbr0,virbr1,virbr0_nic,virbr1_nic,wlp4s0",\n  "ipaddress_docker0": "172.17.0.1",\n  "macaddress_docker0": "02:42:ea:15:d8:84",\n  "netmask_docker0": "255.255.0.0",\n  "mtu_docker0": 1500,\n  "macaddress_em1": "3c:97:0e:e9:28:8e",\n  "mtu_em1": 1500,\n  "ipaddress_lo": "127.0.0.1",\n  "netmask_lo": "255.0.0.0",\n  "mtu_lo": 65536,\n  "macaddress_vethf20ff12": "ae:6e:2b:1e:a1:31",\n  "mtu_vethf20ff12": 1500,\n  "ipaddress_virbr0": "192.168.137.1",\n  "macaddress_virbr0": "52:54:00:ce:82:5e",\n  "netmask_virbr0": "255.255.255.0",\n  "mtu_virbr0": 1500,\n  "ipaddress_virbr1": "192.168.121.1",\n  "macaddress_virbr1": "52:54:00:b4:68:a9",\n  "netmask_virbr1": "255.255.255.0",\n  "mtu_virbr1": 1500,\n  "macaddress_virbr0_nic": "52:54:00:ce:82:5e",\n  "mtu_virbr0_nic": 1500,\n  "macaddress_virbr1_nic": "52:54:00:b4:68:a9",\n  "mtu_virbr1_nic": 1500,\n  "ipaddress_wlp4s0": "192.168.1.19",\n  "macaddress_wlp4s0": "5c:51:4f:e6:a8:e3",\n  "netmask_wlp4s0": "255.255.255.0",\n  "mtu_wlp4s0": 1500,\n  "virtual": "physical",\n  "is_virtual": false,\n  "partitions": {\n    "sda2": {\n      "size": "499091456"\n    },\n    "sda1": {\n      "uuid": "32caaec3-ef40-4691-a3b6-438c3f9bc1c0",\n      "size": "1024000",\n      "mount": "/boot"\n    }\n  },\n  "lsbdistcodename": "TwentyFive",\n  "lsbrelease": ":core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch",  # noqa\n  "filesystems": "btrfs,ext2,ext3,ext4,xfs",\n  "system_uptime": {\n    "seconds": 1558090,\n    "hours": 432,\n    "days": 18,\n    "uptime": "18 days"\n  },\n  "ipaddress": "172.17.0.1",\n  "timezone": "EDT",\n  "ps": "ps -ef",\n  "rubyplatform": "x86_64-linux",\n  "rubysitedir": "/usr/local/share/ruby/site_ruby",\n  "uptime": "18 days",\n  "lsbdistrelease": "25",\n  "operatingsystemrelease": "25",\n  "facterversion": "2.4.3",\n  "kernelrelease": "4.9.14-200.fc25.x86_64",\n  "lsbdistdescription": "Fedora release 25 (Twenty Five)",\n  "network_docker0": "172.17.0.0",\n  "network_lo": "127.0.0.0",\n  "network_virbr0": "192.168.137.0",\n  "network_virbr1": "192.168.121.0",\n  "network_wlp4s0": "192.168.1.0",\n  "lsbdistid": "Fedora",\n  "selinux": true,\n  "selinux_enforced": false,\n  "selinux_policyversion": "30",\n  "selinux_current_mode": "permissive",\n  "selinux_config_mode": "permissive",\n  "selinux_config_policy": "targeted",\n  "hostname": "myhostname",\n  "osfamily": "RedHat",\n  "kernelmajversion": "4.9",\n  "blockdevice_sr0_size": 1073741312,\n  "blockdevice_sr0_vendor": "MATSHITA",\n  "blockdevice_sr0_model": "DVD-RAM UJ8E2",\n  "blockdevice_sda_size": 256060514304,\n  "blockdevice_sda_vendor": "ATA",\n  "blockdevice_sda_model": "SAMSUNG MZ7TD256",\n  "blockdevices": "sda,sr0",\n  "uptime_hours": 432,\n  "kernelversion": "4.9.14"\n}\n'

class TestFacterCollector(BaseFactsTest):
    __test__ = True
    gather_subset = ['!all', 'facter']
    valid_subsets = ['facter']
    fact_namespace = 'ansible_facter'
    collector_class = FacterFactCollector

    def _mock_module(self):
        if False:
            i = 10
            return i + 15
        mock_module = Mock()
        mock_module.params = {'gather_subset': self.gather_subset, 'gather_timeout': 10, 'filter': '*'}
        mock_module.get_bin_path = Mock(return_value='/not/actually/facter')
        mock_module.run_command = Mock(return_value=(0, facter_json_output, ''))
        return mock_module

    @patch('ansible.module_utils.facts.other.facter.FacterFactCollector.get_facter_output')
    def test_bogus_json(self, mock_get_facter_output):
        if False:
            print('Hello World!')
        module = self._mock_module()
        mock_get_facter_output.return_value = '{'
        fact_collector = self.collector_class()
        facts_dict = fact_collector.collect(module=module)
        self.assertIsInstance(facts_dict, dict)
        self.assertEqual(facts_dict, {})

    @patch('ansible.module_utils.facts.other.facter.FacterFactCollector.run_facter')
    def test_facter_non_zero_return_code(self, mock_run_facter):
        if False:
            return 10
        module = self._mock_module()
        mock_run_facter.return_value = (1, '{}', '')
        fact_collector = self.collector_class()
        facts_dict = fact_collector.collect(module=module)
        self.assertIsInstance(facts_dict, dict)
        self.assertNotIn('facter', facts_dict)
        self.assertEqual(facts_dict, {})