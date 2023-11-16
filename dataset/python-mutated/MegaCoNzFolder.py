from pyload.core.network.http.http_request import HTTPRequest
from ..base.decrypter import BaseDecrypter
from ..downloaders.MegaCoNz import MegaClient

class MegaCoNzFolder(BaseDecrypter):
    __name__ = 'MegaCoNzFolder'
    __type__ = 'decrypter'
    __version__ = '0.28'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?mega(?:\\.co)?\\.nz/folder/(?P<ID>[\\w^_]+)#(?P<KEY>[\\w,\\-=]+)(?:/folder/(?P<SUBDIR>[\\w]+))?/?$'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default')]
    __description__ = 'Mega.co.nz folder decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]

    def setup(self):
        if False:
            while True:
                i = 10
        try:
            self.req.http.close()
        except Exception:
            pass
        self.req.http = HTTPRequest(cookies=self.req.cj, options=self.pyload.request_factory.get_options(), limit=10000000)

    def decrypt(self, pyfile):
        if False:
            print('Hello World!')
        id = self.info['pattern']['ID']
        master_key = self.info['pattern']['KEY']
        subdir = self.info['pattern']['SUBDIR']
        self.log_debug('ID: {}'.format(id), 'Key: {}'.format(master_key), 'Type: public folder')
        mega = MegaClient(self, id)
        res = mega.api_request(a='f', c=1, r=1, ca=1, ssl=1)
        if isinstance(res, int):
            mega.check_error(res)
        elif 'e' in res:
            mega.check_error(res['e'])
        urls = ['https://mega.co.nz/folder/{}#{}/file/{}'.format(id, master_key, node['h']) for node in res['f'] if node['t'] == 0 and ':' in node['k'] if subdir is None or node['p'] == subdir]
        if urls:
            self.packages = [(pyfile.package().folder, urls, pyfile.package().name)]