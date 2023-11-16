import pytest
from six import BytesIO
from thefuck.rules.ifconfig_device_not_found import match, get_new_command
from thefuck.types import Command
output = '{}: error fetching interface information: Device not found'
stdout = b'\nwlp2s0    Link encap:Ethernet  HWaddr 5c:51:4f:7c:58:5d\n          inet addr:192.168.0.103  Bcast:192.168.0.255  Mask:255.255.255.0\n          inet6 addr: fe80::be23:69b9:96d2:6d39/64 Scope:Link\n          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1\n          RX packets:23581604 errors:0 dropped:0 overruns:0 frame:0\n          TX packets:17017655 errors:0 dropped:0 overruns:0 carrier:0\n          collisions:0 txqueuelen:1000\n          RX bytes:16148429061 (16.1 GB)  TX bytes:7067533695 (7.0 GB)\n'

@pytest.fixture(autouse=True)
def ifconfig(mocker):
    if False:
        i = 10
        return i + 15
    mock = mocker.patch('thefuck.rules.ifconfig_device_not_found.subprocess.Popen')
    mock.return_value.stdout = BytesIO(stdout)
    return mock

@pytest.mark.parametrize('script, output', [('ifconfig wlan0', output.format('wlan0')), ('ifconfig -s eth0', output.format('eth0'))])
def test_match(script, output):
    if False:
        i = 10
        return i + 15
    assert match(Command(script, output))

@pytest.mark.parametrize('script, output', [('config wlan0', 'wlan0: error fetching interface information: Device not found'), ('ifconfig eth0', '')])
def test_not_match(script, output):
    if False:
        while True:
            i = 10
    assert not match(Command(script, output))

@pytest.mark.parametrize('script, result', [('ifconfig wlan0', ['ifconfig wlp2s0']), ('ifconfig -s wlan0', ['ifconfig -s wlp2s0'])])
def test_get_new_comman(script, result):
    if False:
        while True:
            i = 10
    new_command = get_new_command(Command(script, output.format('wlan0')))
    assert new_command == result