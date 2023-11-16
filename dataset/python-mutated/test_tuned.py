"""
    Test for the salt.modules.tuned
"""
import pytest
from salt.modules import tuned
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {tuned: {}}

def test_v_241():
    if False:
        i = 10
        return i + 15
    '\n    Test the list_ function for older tuned-adm (v2.4.1)\n    as shipped with CentOS-6\n    '
    tuned_list = 'Available profiles:\n- throughput-performance\n- virtual-guest\n- latency-performance\n- laptop-battery-powersave\n- laptop-ac-powersave\n- virtual-host\n- desktop-powersave\n- server-powersave\n- spindown-disk\n- sap\n- enterprise-storage\n- default\nCurrent active profile: throughput-performance'
    mock_cmd = MagicMock(return_value=tuned_list)
    with patch.dict(tuned.__salt__, {'cmd.run': mock_cmd}):
        assert tuned.list_() == ['throughput-performance', 'virtual-guest', 'latency-performance', 'laptop-battery-powersave', 'laptop-ac-powersave', 'virtual-host', 'desktop-powersave', 'server-powersave', 'spindown-disk', 'sap', 'enterprise-storage', 'default']

def test_v_271():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the list_ function for newer tuned-adm (v2.7.1)\n    as shipped with CentOS-7\n    '
    tuned_list = 'Available profiles:\n- balanced                    - General non-specialized tuned profile\n- desktop                     - Optmize for the desktop use-case\n- latency-performance         - Optimize for deterministic performance\n- network-latency             - Optimize for deterministic performance\n- network-throughput          - Optimize for streaming network throughput.\n- powersave                   - Optimize for low power-consumption\n- throughput-performance      - Broadly applicable tuning that provides--\n- virtual-guest               - Optimize for running inside a virtual-guest.\n- virtual-host                - Optimize for running KVM guests\nCurrent active profile: virtual-guest\n'
    mock_cmd = MagicMock(return_value=tuned_list)
    with patch.dict(tuned.__salt__, {'cmd.run': mock_cmd}):
        assert tuned.list_() == ['balanced', 'desktop', 'latency-performance', 'network-latency', 'network-throughput', 'powersave', 'throughput-performance', 'virtual-guest', 'virtual-host']

def test_v_2110_with_warnings():
    if False:
        i = 10
        return i + 15
    '\n    Test the list_ function for newer tuned-adm (v2.11.0)\n    as shipped with CentOS-7.8 when warnings are emitted\n    '
    tuned_list = 'Available profiles:\n- balanced                    - General non-specialized tuned profile\n- desktop                     - Optmize for the desktop use-case\n- latency-performance         - Optimize for deterministic performance\n- network-latency             - Optimize for deterministic performance\n- network-throughput          - Optimize for streaming network throughput.\n- powersave                   - Optimize for low power-consumption\n- throughput-performance      - Broadly applicable tuning that provides--\n- virtual-guest               - Optimize for running inside a virtual-guest.\n- virtual-host                - Optimize for running KVM guests\nCurrent active profile: virtual-guest\n\n** COLLECTED WARNINGS **\nNo SMBIOS nor DMI entry point found, sorry.\n** END OF WARNINGS **\n'
    mock_cmd = MagicMock(return_value=tuned_list)
    with patch.dict(tuned.__salt__, {'cmd.run': mock_cmd}):
        assert tuned.list_() == ['balanced', 'desktop', 'latency-performance', 'network-latency', 'network-throughput', 'powersave', 'throughput-performance', 'virtual-guest', 'virtual-host']

def test_none():
    if False:
        while True:
            i = 10
    ' '
    ret = {'pid': 12345, 'retcode': 1, 'stderr': 'stderr: Cannot talk to Tuned daemon via DBus. Is Tuned daemon running?', 'stdout': 'No current active profile.'}
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(tuned.__salt__, {'cmd.run_all': mock_cmd}):
        assert tuned.active() == 'none'