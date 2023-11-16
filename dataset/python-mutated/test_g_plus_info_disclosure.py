from routersploit.modules.exploits.routers.belkin.g_plus_info_disclosure import Exploit

def test_check_success(target):
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/SaveCfgFile.cgi', methods=['GET'])
    route_mock.return_value = 'testpppoe_usernamepppoe_passwordwl0_pskkeywl0_key1mradius_passwordmradius_secrethttpd_passwordhttp_passwdpppoe_passwdtest'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None