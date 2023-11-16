from flask import request
from routersploit.modules.exploits.routers.dlink.dir_8xx_password_disclosure import Exploit

def apply_response():
    if False:
        while True:
            i = 10
    if 'A' not in request.args.keys():
        response = '\n<?xml version="1.0" encoding="utf-8"?>\n<postxml>\n    <result>FAILED</result>\n    <message>Not authorized</message>\n</postxml>\n    '
    else:
        response = '\n<?xml version="1.0" encoding="utf-8"?>\n<postxml>\n<module>\n    <service>DEVICE.ACCOUNT</service>\n    <device>\n        <account>\n            <seqno></seqno>\n            <max>2</max>\n            <count>1</count>\n            <entry>\n                <uid></uid>\n                <name>Admin</name>\n                <usrid></usrid>\n                <password>RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR</password>\n                <group>0</group>\n                <description></description>\n            </entry>\n        </account>\n        <group>\n            <seqno></seqno>\n            <max></max>\n            <count>0</count>\n        </group>\n        <session>\n            <captcha>0</captcha>\n            <dummy></dummy>\n            <timeout>300</timeout>\n            <maxsession>128</maxsession>\n            <maxauthorized>16</maxauthorized>\n        </session>\n    </device>\n</module>\n</postxml>\n'
    return (response, 200)

def test_exploit_success(target):
    if False:
        while True:
            i = 10
    ' Test scenario - successful exploitation '
    cgi_mock = target.get_route_mock('/getcfg.php', methods=['GET', 'POST'])
    cgi_mock.side_effect = apply_response
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 80
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None