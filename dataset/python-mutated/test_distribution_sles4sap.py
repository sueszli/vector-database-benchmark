from __future__ import annotations
import pytest
from ansible.module_utils.facts.system.distribution import DistributionFiles

@pytest.mark.parametrize('realpath', ('SUSE_SLES_SAP.prod', 'SLES_SAP.prod'))
def test_distribution_sles4sap_suse_sles_sap(mock_module, mocker, realpath):
    if False:
        i = 10
        return i + 15
    mocker.patch('os.path.islink', return_value=True)
    mocker.patch('os.path.realpath', return_value='/etc/products.d/' + realpath)
    test_input = {'name': 'SUSE', 'path': '', 'data': 'suse', 'collected_facts': None}
    test_result = (True, {'distribution': 'SLES_SAP'})
    distribution = DistributionFiles(module=mock_module())
    assert test_result == distribution.parse_distribution_file_SUSE(**test_input)