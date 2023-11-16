from __future__ import annotations
import os
import pytest
from ansible.module_utils.facts.system.distribution import DistributionFiles

@pytest.mark.parametrize(('distro_file', 'expected_version'), (('Slackware', '14.1'), ('SlackwareCurrent', '14.2+')))
def test_parse_distribution_file_slackware(mock_module, distro_file, expected_version):
    if False:
        i = 10
        return i + 15
    with open(os.path.join(os.path.dirname(__file__), '../../fixtures/distribution_files', distro_file)) as file:
        data = file.read()
    test_input = {'name': 'Slackware', 'data': data, 'path': '/etc/os-release', 'collected_facts': None}
    result = (True, {'distribution': 'Slackware', 'distribution_version': expected_version})
    distribution = DistributionFiles(module=mock_module())
    assert result == distribution.parse_distribution_file_Slackware(**test_input)