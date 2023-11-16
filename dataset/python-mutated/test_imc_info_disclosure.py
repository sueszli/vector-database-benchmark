from routersploit.core.exploit.utils import import_exploit
Exploit = import_exploit('routersploit.modules.exploits.routers.3com.imc_info_disclosure')

def test_check_success(target):
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - successful exploitation '
    route_mock = target.get_route_mock('/imc/reportscript/sqlserver/deploypara.properties', methods=['GET'])
    route_mock.return_value = 'TESTreport.db.server.name=ABCDTEST'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 8080
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None