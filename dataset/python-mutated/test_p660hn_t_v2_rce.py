from unittest import mock
from routersploit.modules.exploits.routers.zyxel.p660hn_t_v2_rce import Exploit

@mock.patch('routersploit.modules.exploits.routers.zyxel.p660hn_t_v2_rce.shell')
def test_check_success(mocked_shell, target):
    if False:
        while True:
            i = 10
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/js/Multi_Language.js', methods=['GET'])
    route_mock.return_value = 'TESTP-660HN-T1A_IPv6TEST'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    assert exploit.username == 'supervisor'
    assert exploit.password == 'zyad1234'
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None