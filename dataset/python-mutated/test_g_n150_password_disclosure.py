from routersploit.modules.exploits.routers.belkin.g_n150_password_disclosure import Exploit

def test_exploit_success(target):
    if False:
        i = 10
        return i + 15
    ' Test scenario - successful exploitation '
    route_mock = target.get_route_mock('/login.stm', methods=['GET'])
    route_mock.return_value = 'testpassword= "admin1234"test'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None