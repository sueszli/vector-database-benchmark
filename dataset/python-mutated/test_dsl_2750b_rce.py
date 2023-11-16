from unittest import mock
from routersploit.modules.exploits.routers.dlink.dsl_2750b_rce import Exploit

@mock.patch('routersploit.modules.exploits.routers.dlink.dsl_2750b_rce.shell')
def test_check_success(mocked_shell, target):
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - successful exploitation '
    route_mock1 = target.get_route_mock('/login.cgi', methods=['GET'])
    route_mock1.return_value = 'TEST'
    route_mock2 = target.get_route_mock('/ayefeaturesconvert.js', methods=['GET'])
    route_mock2.return_value = '\n        (..)\n        var AYECOM_PRIVATE="private";\n        var AYECOM_AREA="EU";\n        var AYECOM_FWVER="1.01";\n        var AYECOM_HWVER="D1";\n        var AYECOM_PRIVATEDIR="private";\n        var AYECOM_PROFILE="DSL-2750B";\n        var FIRST_HTML="";\n        var BUILD_GUI_VERSIOIN_EU="y";\n        // BUILD_GUI_VERSIOIN_AU is not s\n        (..)\n        '
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None