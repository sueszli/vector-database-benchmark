from routersploit.modules.exploits.routers.dlink.dsl_2750b_info_disclosure import Exploit

def test_check_success(target):
    if False:
        i = 10
        return i + 15
    ' Test scenario - successful exploitation '
    cgi_mock = target.get_route_mock('/hidden_info.html', methods=['GET'])
    cgi_mock.return_value = 'TESTPassPhraseTESTSSIDTEST'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None