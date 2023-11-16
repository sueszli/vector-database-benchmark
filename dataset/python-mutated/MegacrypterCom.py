import json
import re
from .MegaCoNz import MegaCoNz, MegaCrypto

class MegacrypterCom(MegaCoNz):
    __name__ = 'MegacrypterCom'
    __type__ = 'downloader'
    __version__ = '0.28'
    __status__ = 'testing'
    __pattern__ = 'https?://\\w{0,10}\\.?megacrypter\\.com/[\\w\\-!]+'
    __config__ = [('enabled', 'bool', 'Activated', True)]
    __description__ = 'Megacrypter.com decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GonzaloSR', 'gonzalo@gonzalosr.com')]
    API_URL = 'http://megacrypter.com/api'
    FILE_SUFFIX = '.crypted'

    def api_request(self, **kwargs):
        if False:
            return 10
        '\n        Dispatch a call to the api, see megacrypter.com/api_doc.\n        '
        self.log_debug('JSON request: ' + json.dumps(kwargs))
        res = self.load(self.API_URL, post=json.dumps(kwargs))
        self.log_debug('API Response: ' + res)
        return json.loads(res)

    def process(self, pyfile):
        if False:
            print('Hello World!')
        node = re.match(self.__pattern__, pyfile.url).group(0)
        info = self.api_request(link=node, m='info')
        dl = self.api_request(link=node, m='dl')
        key = MegaCrypto.base64_decode(info['key'])
        pyfile.name = info['name'] + self.FILE_SUFFIX
        self.download(dl['url'])
        self.decrypt_file(key)
        pyfile.name = info['name']