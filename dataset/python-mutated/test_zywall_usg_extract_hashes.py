from routersploit.modules.exploits.routers.zyxel.zywall_usg_extract_hashes import Exploit

def test_check_success(target):
    if False:
        i = 10
        return i + 15
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/cgi-bin/export-cgi/images/', methods=['GET'])
    route_mock.return_value = 'TEST\nusername TEST password TEST user-type TEST\nTEST\n'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 443
    assert exploit.ssl is True
    exploit.target = target.host
    exploit.port = target.port
    exploit.ssl = 'false'
    assert exploit.check()
    assert exploit.run() is None