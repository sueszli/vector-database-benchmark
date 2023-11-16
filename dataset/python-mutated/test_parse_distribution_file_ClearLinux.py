from __future__ import annotations
import os
import pytest
from ansible.module_utils.facts.system.distribution import DistributionFiles

@pytest.fixture
def test_input():
    if False:
        print('Hello World!')
    return {'name': 'Clearlinux', 'path': '/usr/lib/os-release', 'collected_facts': None}

def test_parse_distribution_file_clear_linux(mock_module, test_input):
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(os.path.dirname(__file__), '../../fixtures/distribution_files/ClearLinux')) as file:
        test_input['data'] = file.read()
    result = (True, {'distribution': 'Clear Linux OS', 'distribution_major_version': '28120', 'distribution_release': 'clear-linux-os', 'distribution_version': '28120'})
    distribution = DistributionFiles(module=mock_module())
    assert result == distribution.parse_distribution_file_ClearLinux(**test_input)

@pytest.mark.parametrize('distro_file', ('CoreOS', 'LinuxMint'))
def test_parse_distribution_file_clear_linux_no_match(mock_module, distro_file, test_input):
    if False:
        while True:
            i = 10
    '\n    Test against data from Linux Mint and CoreOS to ensure we do not get a reported\n    match from parse_distribution_file_ClearLinux()\n    '
    with open(os.path.join(os.path.dirname(__file__), '../../fixtures/distribution_files', distro_file)) as file:
        test_input['data'] = file.read()
    result = (False, {})
    distribution = DistributionFiles(module=mock_module())
    assert result == distribution.parse_distribution_file_ClearLinux(**test_input)