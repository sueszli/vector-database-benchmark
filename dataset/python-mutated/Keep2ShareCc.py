import json
import re
from pyload.core.network.http.exceptions import BadHeader
from ..base.simple_downloader import SimpleDownloader

class Keep2ShareCc(SimpleDownloader):
    __name__ = 'Keep2ShareCc'
    __type__ = 'downloader'
    __version__ = '0.47'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?(keep2share|k2s|keep2s)\\.cc/file/(?P<ID>\\w+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Keep2Share.cc downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('stickell', 'l.stickell@yahoo.it'), ('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    URL_REPLACEMENTS = [(__pattern__ + '.*', 'https://k2s.cc/file/\\g<ID>')]
    API_URL = 'https://keep2share.cc/api/v2/'

    def api_request(self, method, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        html = self.load(self.API_URL + method, post=json.dumps(kwargs))
        return json.loads(html)

    def api_info(self, url):
        if False:
            print('Hello World!')
        file_id = re.match(self.__pattern__, url).group('ID')
        file_info = self.api_request('GetFilesInfo', ids=[file_id], extended_info=False)
        if file_info['code'] != 200 or len(file_info['files']) == 0 or (not file_info['files'][0].get('is_available')):
            return {'status': 1}
        else:
            return {'name': file_info['files'][0]['name'], 'size': file_info['files'][0]['size'], 'md5': file_info['files'][0]['md5'], 'access': file_info['files'][0]['access'], 'free_access': file_info['files'][0]['isAvailableForFree'], 'status': 2 if file_info['files'][0]['is_available'] else 1}

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.multi_dl = self.premium
        self.resume_download = True

    def handle_free(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        file_id = self.info['pattern']['ID']
        if self.info['access'] == 'premium' or self.info['free_access'] is False:
            self.fail(self._('File can be downloaded by premium users only'))
        elif self.info['access'] == 'private':
            self.fail(self._('This is a private file'))
        try:
            json_data = self.api_request('GetUrl', file_id=file_id, free_download_key=None, captcha_challenge=None, captcha_response=None)
        except BadHeader as exc:
            if exc.code == 406:
                for i in range(10):
                    json_data = self.api_request('RequestCaptcha')
                    if json_data['code'] != 200:
                        self.fail(self._('Request captcha API failed'))
                    captcha_response = self.captcha.decrypt(json_data['captcha_url'])
                    try:
                        json_data = self.api_request('GetUrl', file_id=file_id, free_download_key=None, captcha_challenge=json_data['challenge'], captcha_response=captcha_response)
                    except BadHeader as exc:
                        if exc.code == 406:
                            json_data = json.loads(exc.content)
                            if json_data['errorCode'] == 31:
                                self.captcha.invalid()
                                continue
                            elif json_data['errorCode'] == 42:
                                self.captcha.correct()
                                self.retry(wait=json_data['errors'][0]['timeRemaining'])
                            else:
                                self.fail(json_data['message'])
                        else:
                            raise
                    else:
                        self.captcha.correct()
                        free_download_key = json_data['free_download_key']
                        break
                else:
                    self.fail(self._('Max captcha retries reached'))
                self.wait(json_data['time_wait'])
                json_data = self.api_request('GetUrl', file_id=file_id, free_download_key=free_download_key, captcha_challenge=None, captcha_response=None)
                if json_data['code'] == 200:
                    self.link = json_data['url']
            else:
                raise
        else:
            self.link = json_data['url']

    def handle_premium(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        file_id = self.info['pattern']['ID']
        if self.info['access'] == 'private':
            self.fail(self._('This is a private file'))
        json_data = self.api_request('GetUrl', file_id=file_id, free_download_key=None, captcha_challenge=None, captcha_response=None, auth_token=self.account.info['data']['token'])
        self.link = json_data['url']