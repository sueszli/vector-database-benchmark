from ssshare import *
import threading
from ssshare.ss import crawler, ssr_check
import requests

def test2():
    if False:
        print('Hello World!')
    data = '\n{\n  "server": "203.104.205.115",\n  "server_ipv6": "::",\n  "server_port": 8080,\n  "local_address": "127.0.0.1",\n  "local_port": 1080,\n  "password": "yui",\n  "group": "Charles Xu",\n  "obfs": "tls1.2_ticket_auth",\n  "method": "chacha20",\n  "ssr_protocol": "auth_sha1_v4",\n  "obfsparam": "",\n  "protoparam": ""\n}'
    w = ssr_check.test_socks_server(str_json=data)
    print('>>>>>>>结果:', w)
    if w is True:
        print(data)
    elif w == -1:
        print(data)
        raise Exception('sodium test failed')
print('-----------测试：子线程----------')
t = threading.Thread(target=test2)
t.start()
t.join()