from routersploit.modules.creds.routers.mikrotik.api_ros_default_creds import Exploit

def test_check_success(tcp_target):
    if False:
        return 10
    ' Test scenario - testing against mikrotik api ros server '
    exploit = Exploit()
    exploit.target = tcp_target.host
    exploit.port = tcp_target.port
    assert exploit.check()