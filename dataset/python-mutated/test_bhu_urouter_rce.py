from unittest import mock
from routersploit.modules.exploits.routers.bhu.bhu_urouter_rce import Exploit

@mock.patch('routersploit.modules.exploits.routers.bhu.bhu_urouter_rce.shell')
def test_check_success(mocked_shell, target):
    if False:
        return 10
    ' Test scenario - successful check '
    route_mock1 = target.get_route_mock('/cgi-bin/cgiSrv.cgi', methods=['POST'])
    route_mock1.return_value = 'teststatus="doing"test'
    route_mock2 = target.get_route_mock('/routersploit.check', methods=['GET'])
    route_mock2.return_value = 'testroottest'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None