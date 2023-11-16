from routersploit.modules.exploits.routers.cisco.firepower_management60_path_traversal import Exploit

def test_check_success(target):
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - successful exploitation '
    route_mock = target.get_route_mock('/login.cgi', methods=['GET'])
    route_mock.return_value = 'test6.0.1test'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    assert exploit.path == '/etc/passwd'
    assert exploit.username == 'admin'
    assert exploit.password == 'Admin123'
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None