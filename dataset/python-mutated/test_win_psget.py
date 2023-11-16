"""
    Test cases for salt.modules.win_psget
"""
import pytest
import salt.modules.win_psget as win_psget
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {win_psget: {}}

@pytest.fixture
def bootstrap_ps_str():
    if False:
        for i in range(10):
            print('nop')
    return '<?xml version="1.0" encoding="utf-8"?>\n<Objects>\n  <Object Type="System.Management.Automation.PSCustomObject">\n    <Property Name="Name" Type="System.String">NuGet</Property>\n    <Property Name="Version" Type="Microsoft.PackageManagement.Internal.Utility.Versions.FourPartVersion">\n      <Property Name="Major" Type="System.UInt16">2</Property>\n      <Property Name="Minor" Type="System.UInt16">8</Property>\n      <Property Name="Build" Type="System.UInt16">5</Property>\n      <Property Name="Revision" Type="System.UInt16">208</Property>\n    </Property>\n    <Property Name="ProviderPath" Type="System.String">C:\\Program Files\\PackageManagement\\ProviderAssemblies\\nuget\\2.8.5\n.208\\Microsoft.PackageManagement.NuGetProvider.dll</Property>\n  </Object>\n</Objects>'

@pytest.fixture
def avail_modules_ps_str():
    if False:
        while True:
            i = 10
    return '<?xml version="1.0" encoding="utf-8"?>\n<Objects>\n  <Object Type="System.Management.Automation.PSCustomObject">\n    <Property Name="Name" Type="System.String">ActOnCmdlets</Property>\n    <Property Name="Description" Type="System.String">CData Cmdlets for Act-On</Property>\n  </Object>\n  <Object Type="System.Management.Automation.PSCustomObject">\n    <Property Name="Name" Type="System.String">FinancialEdgeNXTCmdlets</Property>\n    <Property Name="Description" Type="System.String">CData Cmdlets for Blackbaud Financial Edge NXT</Property>\n  </Object>\n  <Object Type="System.Management.Automation.PSCustomObject">\n    <Property Name="Name" Type="System.String">GoogleCMCmdlets</Property>\n    <Property Name="Description" Type="System.String">CData Cmdlets for Google Campaign Manager</Property>\n  </Object>\n  <Object Type="System.Management.Automation.PSCustomObject">\n    <Property Name="Name" Type="System.String">DHCPMigration</Property>\n    <Property Name="Description" Type="System.String">A module to copy various DHCP information from 1 server to another.</Property>\n  </Object>\n</Objects>'

def test_bootstrap(bootstrap_ps_str):
    if False:
        i = 10
        return i + 15
    mock_read_ok = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': bootstrap_ps_str})
    with patch.dict(win_psget.__salt__, {'cmd.run_all': mock_read_ok}):
        assert 'NuGet' in win_psget.bootstrap()

def test_avail_modules(avail_modules_ps_str):
    if False:
        return 10
    mock_read_ok = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': avail_modules_ps_str})
    with patch.dict(win_psget.__salt__, {'cmd.run_all': mock_read_ok}):
        assert 'DHCPMigration' in win_psget.avail_modules(False)
        assert 'DHCPMigration' in win_psget.avail_modules(True)