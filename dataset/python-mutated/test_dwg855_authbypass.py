from routersploit.modules.exploits.routers.technicolor.dwg855_authbypass import Exploit

def test_check_success(target):
    if False:
        i = 10
        return i + 15
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/logo.jpg', methods=['GET'])
    route_mock.return_value = b'\x11Ducky\x00'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    assert exploit.nuser == 'ruser'
    assert exploit.npass == 'rpass'
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None