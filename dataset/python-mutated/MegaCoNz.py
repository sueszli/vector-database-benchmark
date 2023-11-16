import base64
import json
import os
import random
import re
import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from pyload.core.network.http.exceptions import BadHeader
from pyload.core.utils.convert import to_bytes
from ..base.downloader import BaseDownloader
from ..helpers import exists

class MegaCrypto:

    @staticmethod
    def base64_decode(data):
        if False:
            return 10
        data = to_bytes(data, 'ascii')
        data += b'=' * (-len(data) % 4)
        return base64.b64decode(data, b'-_')

    @staticmethod
    def base64_encode(data):
        if False:
            while True:
                i = 10
        return base64.b64encode(data, b'-_')

    @staticmethod
    def a32_to_bytes(a):
        if False:
            print('Hello World!')
        return struct.pack('>{}I'.format(len(a)), *a)

    @staticmethod
    def bytes_to_a32(s):
        if False:
            print('Hello World!')
        s += b'\x00' * (-len(s) % 4)
        return struct.unpack('>{}I'.format(len(s) // 4), s)

    @staticmethod
    def a32_to_base64(a):
        if False:
            print('Hello World!')
        return MegaCrypto.base64_encode(MegaCrypto.a32_to_bytes(a))

    @staticmethod
    def base64_to_a32(s):
        if False:
            print('Hello World!')
        return MegaCrypto.bytes_to_a32(MegaCrypto.base64_decode(s))

    @staticmethod
    def cbc_decrypt(data, key):
        if False:
            i = 10
            return i + 15
        cipher = Cipher(algorithms.AES(MegaCrypto.a32_to_bytes(key)), modes.CBC(b'\x00' * 16), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(data) + decryptor.finalize()

    @staticmethod
    def cbc_encrypt(data, key):
        if False:
            print('Hello World!')
        cipher = Cipher(algorithms.AES(MegaCrypto.a32_to_bytes(key)), modes.CBC(b'\x00' * 16), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    @staticmethod
    def ecb_decrypt(data, key):
        if False:
            while True:
                i = 10
        cipher = Cipher(algorithms.AES(MegaCrypto.a32_to_bytes(key)), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(data) + decryptor.finalize()

    @staticmethod
    def ecb_encrypt(data, key):
        if False:
            while True:
                i = 10
        cipher = Cipher(algorithms.AES(MegaCrypto.a32_to_bytes(key)), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(data) + encryptor.finalize()

    @staticmethod
    def get_cipher_key(key):
        if False:
            while True:
                i = 10
        '\n        Construct the cipher key from the given data.\n        '
        k = (key[0] ^ key[4], key[1] ^ key[5], key[2] ^ key[6], key[3] ^ key[7])
        iv = key[4:6] + (0, 0)
        meta_mac = key[6:8]
        return (k, iv, meta_mac)

    @staticmethod
    def decrypt_attr(data, key):
        if False:
            for i in range(10):
                print('nop')
        "\n        Decrypt an encrypted attribute (usually 'a' or 'at' member of a node)\n        "
        data = MegaCrypto.base64_decode(data)
        (k, iv, meta_mac) = MegaCrypto.get_cipher_key(key)
        attr = MegaCrypto.cbc_decrypt(data, k)
        return json.loads(re.search(b'{.+}', attr).group(0)) if attr[:6] == b'MEGA{"' else False

    @staticmethod
    def decrypt_key(data, key):
        if False:
            return 10
        "\n        Decrypt an encrypted key ('k' member of a node)\n        "
        data = MegaCrypto.base64_decode(data)
        return MegaCrypto.bytes_to_a32(MegaCrypto.ecb_decrypt(data, key))

    @staticmethod
    def encrypt_key(data, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Encrypt a decrypted key.\n        '
        data = MegaCrypto.a32_to_bytes(data)
        return MegaCrypto.bytes_to_a32(MegaCrypto.ecb_encrypt(data, key))

    @staticmethod
    def get_chunks(size):
        if False:
            return 10
        '\n        Calculate chunks for a given encrypted file size.\n        '
        chunk_start = 0
        chunk_size = 131072
        while chunk_start + chunk_size < size:
            yield (chunk_start, chunk_size)
            chunk_start += chunk_size
            if chunk_size < 1048576:
                chunk_size += 131072
        if chunk_start < size:
            yield (chunk_start, size - chunk_start)

    class Checksum:
        """
        interface for checking CBC-MAC checksum.
        """

        def __init__(self, key):
            if False:
                while True:
                    i = 10
            (k, iv, meta_mac) = MegaCrypto.get_cipher_key(key)
            self.hash = b'\x00' * 16
            self.key = MegaCrypto.a32_to_bytes(k)
            self.iv = MegaCrypto.a32_to_bytes(iv[0:2] * 2)
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.hash), backend=default_backend())
            self.AES = cipher.encryptor()

        def update(self, chunk):
            if False:
                i = 10
                return i + 15
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())
            encryptor = cipher.encryptor()
            for j in range(0, len(chunk), 16):
                block = chunk[j:j + 16].ljust(16, b'\x00')
                hash = encryptor.update(block)
            encryptor.finalize()
            self.hash = self.AES.update(hash)

        def digest(self):
            if False:
                while True:
                    i = 10
            '\n            Return the **binary** (non-printable) CBC-MAC of the message that has been\n            authenticated so far.\n            '
            d = MegaCrypto.bytes_to_a32(self.hash)
            return (d[0] ^ d[1], d[2] ^ d[3])

        def hexdigest(self):
            if False:
                while True:
                    i = 10
            '\n            Return the **printable** CBC-MAC of the message that has been authenticated\n            so far.\n            '
            return ''.join(('{:2x}'.format(ord(x)) for x in MegaCrypto.a32_to_bytes(self.digest())))

        @staticmethod
        def new(key):
            if False:
                return 10
            return MegaCrypto.Checksum(key)

class MegaClient:
    API_URL = 'https://eu.api.mega.co.nz/cs'

    def __init__(self, plugin, node_id):
        if False:
            i = 10
            return i + 15
        self.plugin = plugin
        self._ = plugin._
        self.node_id = node_id

    def api_request(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Dispatch a call to the api, see https://mega.co.nz/#developers.\n        '
        uid = random.randint(10 << 9, 10 ** 10)
        get_params = {'id': uid}
        if self.node_id:
            get_params['n'] = self.node_id
        if hasattr(self.plugin, 'account'):
            if self.plugin.account:
                mega_session_id = self.plugin.account.info['data'].get('mega_session_id', None)
            else:
                mega_session_id = None
        else:
            mega_session_id = self.plugin.info['data'].get('mega_session_id', None)
        if mega_session_id:
            get_params['sid'] = mega_session_id
        try:
            res = self.plugin.load(self.API_URL, get=get_params, post=json.dumps([kwargs]))
        except BadHeader as exc:
            if exc.code == 500:
                self.plugin.retry(wait_time=60, reason=self._('Server busy'))
            else:
                raise
        self.plugin.log_debug('Api Response: ' + res)
        res = json.loads(res)
        if isinstance(res, list):
            res = res[0]
        return res

    def check_error(self, code):
        if False:
            while True:
                i = 10
        ecode = abs(code)
        if ecode in (9, 16, 21):
            self.plugin.offline()
        elif ecode in (3, 13, 17, 18, 19, 24):
            self.plugin.temp_offline()
        elif ecode in (1, 4, 6, 10, 15):
            self.plugin.retry(max_tries=5, wait_time=30, reason=self._('Error code: [{}]').format(-ecode))
        else:
            self.plugin.fail(self._('Error code: [{}]').format(-ecode))

class MegaCoNz(BaseDownloader):
    __name__ = 'MegaCoNz'
    __type__ = 'downloader'
    __version__ = '0.58'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?mega(?:\\.co)?\\.nz/(?:file/(?P<ID1>[\\w^_]+)#(?P<K1>[\\w\\-,=]+)|folder/(?P<ID2>[\\w^_]+)#(?P<K2>[\\w\\-,=]+)/file/(?P<NID>[\\w^_]+)|#!(?P<ID3>[\\w^_]+)!(?P<K3>[\\w\\-,=]+))'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Mega.co.nz downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('RaNaN', 'ranan@pyload.net'), ('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT}yahoo[DOT]com')]
    FILE_SUFFIX = '.crypted'

    def decrypt_file(self, key):
        if False:
            print('Hello World!')
        "\n        Decrypts and verifies checksum to the file at 'last_download'.\n        "
        (k, iv, meta_mac) = MegaCrypto.get_cipher_key(key)
        cipher = Cipher(algorithms.AES(MegaCrypto.a32_to_bytes(k)), modes.CTR(MegaCrypto.a32_to_bytes(iv)), backend=default_backend())
        decryptor = cipher.decryptor()
        self.pyfile.set_status('decrypting')
        self.pyfile.set_progress(0)
        file_crypted = os.fsdecode(self.last_download)
        file_decrypted = file_crypted.rsplit(self.FILE_SUFFIX)[0]
        try:
            f = open(file_crypted, mode='rb')
            df = open(file_decrypted, mode='wb')
        except IOError as exc:
            self.fail(exc)
        encrypted_size = os.path.getsize(file_crypted)
        checksum_activated = self.config.get('enabled', default=False, plugin='Checksum')
        check_checksum = self.config.get('check_checksum', default=True, plugin='Checksum')
        cbc_mac = MegaCrypto.Checksum(key) if checksum_activated and check_checksum else None
        progress = 0
        for (chunk_start, chunk_size) in MegaCrypto.get_chunks(encrypted_size):
            buf = f.read(chunk_size)
            if not buf:
                break
            chunk = decryptor.update(buf)
            df.write(chunk)
            progress += chunk_size
            self.pyfile.set_progress(100 * progress // encrypted_size)
            if checksum_activated and check_checksum:
                cbc_mac.update(chunk)
        df.write(decryptor.finalize())
        self.pyfile.set_progress(100)
        f.close()
        df.close()
        self.log_info(self._('File decrypted'))
        os.remove(file_crypted)
        if checksum_activated and check_checksum:
            file_mac = cbc_mac.digest()
            if file_mac == meta_mac:
                self.log_info(self._('File integrity of "{}" verified by CBC-MAC checksum ({})').format(self.pyfile.name.rsplit(self.FILE_SUFFIX)[0], meta_mac))
            else:
                self.log_warning(self._('CBC-MAC checksum for file "{}" does not match ({} != {})').format(self.pyfile.name.rsplit(self.FILE_SUFFIX)[0], file_mac, meta_mac))
                self.checksum_failed(file_decrypted, self._('Checksums do not match'))
        self.last_download = file_decrypted

    def checksum_failed(self, local_file, msg):
        if False:
            while True:
                i = 10
        check_action = self.config.get('check_action', default='retry', plugin='Checksum')
        if check_action == 'retry':
            max_tries = self.config.get('max_tries', default=2, plugin='Checksum')
            retry_action = self.config.get('retry_action', default='fail', plugin='Checksum')
            if all((r < max_tries for (_, r) in self.retries.items())):
                os.remove(local_file)
                wait_time = self.config.get('wait_time', default=1, plugin='Checksum')
                self.retry(max_tries, wait_time, msg)
            elif retry_action == 'nothing':
                return
        elif check_action == 'nothing':
            return
        os.remove(local_file)
        self.fail(msg)

    def check_exists(self, name):
        if False:
            print('Hello World!')
        "\n        Because of Mega downloads a temporary encrypted file with the extension of\n        '.crypted', pyLoad cannot correctly detect if the file exists before\n        downloading. This function corrects this.\n\n        Raises Skip() if file exists and 'skip_existing' configuration option is\n        set to True.\n        "
        if self.pyload.config.get('download', 'skip_existing'):
            storage_folder = self.pyload.config.get('general', 'storage_folder')
            dest_file = os.path.join(storage_folder, self.pyfile.package().folder if self.pyload.config.get('general', 'folder_per_package') else '', name)
            if exists(dest_file):
                self.pyfile.name = name
                self.skip(self._('File exists.'))

    def process(self, pyfile):
        if False:
            print('Hello World!')
        node_id = self.info['pattern']['NID']
        public = node_id in ('', None)
        id = self.info['pattern']['ID1'] or self.info['pattern']['ID2'] or self.info['pattern']['ID3']
        key = self.info['pattern']['K1'] or self.info['pattern']['K2'] or self.info['pattern']['K3']
        self.log_debug('ID: {},'.format(id), 'Key: {}'.format(key), 'Type: {}'.format('public' if public else 'node'), 'Owner: {}'.format(node_id))
        mega = MegaClient(self, id)
        master_key = MegaCrypto.base64_to_a32(key)
        if not public:
            res = mega.api_request(a='f', c=1, r=1, ca=1, ssl=1)
            if isinstance(res, int):
                mega.check_error(res)
            elif isinstance(res, dict) and 'e' in res:
                mega.check_error(res['e'])
            for node in res['f']:
                if node['t'] == 0 and ':' in node['k'] and (node['h'] == node_id):
                    master_key = MegaCrypto.decrypt_key(node['k'][node['k'].index(':') + 1:], master_key)
                    break
            else:
                self.offline()
        if len(master_key) != 8:
            self.log_error(self._('Invalid key length'))
            self.fail(self._('Invalid key length'))
        if public:
            res = mega.api_request(a='g', g=1, p=id, ssl=1)
        else:
            res = mega.api_request(a='g', g=1, n=node_id, ssl=1)
        if isinstance(res, int):
            mega.check_error(res)
        elif isinstance(res, dict) and 'e' in res:
            mega.check_error(res['e'])
        attr = MegaCrypto.decrypt_attr(res['at'], master_key)
        if not attr:
            self.fail(self._('Decryption failed'))
        self.log_debug(f'Decrypted Attr: {attr}')
        name = attr['n']
        self.check_exists(name)
        pyfile.name = name + self.FILE_SUFFIX
        pyfile.size = res['s']
        time_left = res.get('tl', 0)
        if time_left:
            self.log_warning(self._('Free download limit reached'))
            self.retry(wait=time_left, msg=self._('Free download limit reached'))
        try:
            self.download(res['g'], disposition=False)
        except BadHeader as exc:
            if exc.code == 509:
                self.fail(self._('Bandwidth Limit Exceeded'))
            else:
                raise
        self.decrypt_file(master_key)
        pyfile.name = name