from flask import Response
from routersploit.modules.exploits.cameras.jovision.jovision_credentials_disclosure import Exploit

def apply_response(*args, **kwargs):
    if False:
        print('Hello World!')
    response = '\n        [{\n            "nIndex":\t0,\n            "acID":\t"admin",\n            "acPW":\t"admin1234",\n            "acDescript":\t"admin account",\n            "nPower":\t20\n        }]\n        '
    resp = Response(response, status=200)
    resp.headers['Content-Type'] = 'application/json'
    return resp

def test_check_success(target):
    if False:
        for i in range(10):
            print('nop')
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/cgi-bin/jvsweb.cgi', methods=['GET'])
    route_mock.side_effect = apply_response
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check() is True
    assert exploit.run() is None