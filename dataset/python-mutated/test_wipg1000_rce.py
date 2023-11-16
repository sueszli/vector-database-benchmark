from unittest import mock
from routersploit.modules.exploits.misc.wepresent.wipg1000_rce import Exploit

@mock.patch('routersploit.modules.exploits.misc.wepresent.wipg1000_rce.shell')
def test_check_success(mocked_shell, target):
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - successful check '
    cgi_mock = target.get_route_mock('/cgi-bin/rdfs.cgi', methods=['GET'])
    cgi_mock.return_value = 'testFollow administrator instructions to enter the complete pathtest'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None