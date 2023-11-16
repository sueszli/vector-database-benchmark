from routersploit.modules.exploits.routers.dlink.dvg_n5402sp_path_traversal import Exploit

def test_check_success(target):
    if False:
        return 10
    ' Test scenario - successful check '
    route_mock = target.get_route_mock('/cgi-bin/webproc', methods=['POST'])
    route_mock.return_value = 'root:x:0:0:root:/root:/bin/bashdaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologinbin:x:2:2:bin:/bin:/usr/sbin/nologinsys:x:3:3:sys:/dev:/usr/sbin/nologinsync:x:4:65534:sync:/bin:/bin/syncgames:x:5:60:games:/usr/games:/usr/sbin/nologinman:x:6:12:man:/var/cache/man:/usr/sbin/nologinlp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologinmail:x:8:8:mail:/var/mail:/usr/sbin/nologinnews:x:9:9:news:/var/spool/news:/usr/sbin/nologinuucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologinproxy:x:13:13:proxy:/bin:/usr/sbin/nologinwww-data:x:33:33:www-data:/var/www:/usr/sbin/nologinbackup:x:34:34:backup:/var/backups:/usr/sbin/nologin'
    exploit = Exploit()
    assert exploit.target == ''
    assert exploit.port == 8080
    assert exploit.filename == '/etc/shadow'
    exploit.target = target.host
    exploit.port = target.port
    assert exploit.check()
    assert exploit.run() is None