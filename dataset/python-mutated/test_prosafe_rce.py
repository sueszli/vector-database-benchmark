from unittest import mock
from flask import request
from routersploit.modules.exploits.routers.netgear.prosafe_rce import Exploit

def apply_response(*args, **kwargs):
    if False:
        while True:
            i = 10
    res = request.form['reqMethod']
    data = 'TEST' + res + 'TEST'
    return (data, 200)

@mock.patch('routersploit.modules.exploits.routers.netgear.prosafe_rce.shell')
def test_exploit_success(mocked_shell, target):
    if False:
        i = 10
        return i + 15
    ' Test scenario - successful exploitation '
    route_mock = target.get_route_mock('/login_handler.php', methods=['POST'])
    route_mock.side_effect = apply_response
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None