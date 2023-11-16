from routersploit.modules.exploits.routers.dlink.dir_645_password_disclosure import Exploit

def test_check_success(target):
    if False:
        print('Hello World!')
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/getcfg.php', methods=['POST'])
    route_mock.return_value = '\n        <?xml version="1.0" encoding="utf-8"?>\n        <postxml>\n        <module>\n            <service>DEVICE.ACCOUNT</service>\n            <device>\n                <gw_name>DIR-645</gw_name>\n\n                <account>\n                    <seqno>2</seqno>\n                    <max>2</max>\n                    <count>2</count>\n                    <entry>\n                        <uid>USR-</uid>\n                        <name>admin</name>\n                        <usrid></usrid>\n                        <password>0920983386</password>\n                        <group>0</group>\n                        <description></description>\n                    </entry>\n                    <entry>\n                        <uid>USR-1</uid>\n                        <name>user</name>\n                        <usrid></usrid>\n                        <password>3616441</password>\n                        <group>101</group>\n                        <description></description>\n                    </entry>\n                </account>\n                <group>\n                    <seqno></seqno>\n                    <max></max>\n                    <count>0</count>\n                </group>\n                <session>\n                    <captcha>0</captcha>\n                    <dummy></dummy>\n                    <timeout>600</timeout>\n                    <maxsession>128</maxsession>\n                    <maxauthorized>16</maxauthorized>\n                </session>\n            </device>\n        </module>\n        </postxml>\n        '
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 8080
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None