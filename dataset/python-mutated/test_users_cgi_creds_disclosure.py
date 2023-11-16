from flask import request, Response
from base64 import b64decode
from routersploit.modules.exploits.cameras.brickcom.users_cgi_creds_disclosure import Exploit
response = '\n    size=4\n    User1.index=1\n    User1.username=admin\n    User1.password=test1234\n    User1.privilege=1\n\n    User2.index=2\n    User2.username=viewer\n    User2.password=viewer\n    User2.privilege=0\n\n    User3.index=3\n    User3.username=rviewer\n    User3.password=rviewer\n    User3.privilege=2\n\n    User4.index=0\n    User4.username=visual\n    User4.password=visual1234\n    User4.privilege=0\n    '

def apply_response(*args, **kwargs):
    if False:
        return 10
    if 'Authorization' in request.headers.keys():
        creds = str(b64decode(request.headers['Authorization'].replace('Basic ', '')), 'utf-8')
        if creds in ['rviewer:rviewer']:
            return (response, 200)
    resp = Response('Unauthorized')
    resp.headers['WWW-Authenticate'] = 'Basic ABC'
    return (resp, 401)

def test_check_success(target):
    if False:
        return 10
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/cgi-bin/users.cgi', methods=['GET', 'POST'])
    route_mock.side_effect = apply_response
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check() is True
    assert exploit.run() is None