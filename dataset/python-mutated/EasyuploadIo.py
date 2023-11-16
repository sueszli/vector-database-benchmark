import json
import re
import urllib.parse
from ..anticaptchas.ReCaptcha import ReCaptcha
from ..base.simple_downloader import SimpleDownloader

class EasyuploadIo(SimpleDownloader):
    __name__ = 'EasyuploadIo'
    __type__ = 'downloader'
    __version__ = '0.02'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?easyupload\\.io/(?P<ID>\\w+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Easyupload.io downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = '<h4>(?P<N>.+?)</h4>'
    SIZE_PATTERN = '>Size: (?P<S>[\\d.,]+) (?P<U>[\\w^_]+)'
    OFFLINE_PATTERN = '<h4>FILE NOT FOUND</h4>'
    RECAPTCHA_PATTERN = "grecaptcha.execute\\('((?:[\\w\\-]|%[0-9a-fA-F]{2})+)'"

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.multi_dl = True

    def handle_free(self, pyfile):
        if False:
            while True:
                i = 10
        password_protected = "$('#password-modal').modal('open');" in self.data
        m = re.search('url: "(https://\\w+\\.easyupload\\.io/action\\.php)"', self.data)
        if m is None:
            self.fail('action URL not found')
        else:
            action_url = m.group(1)
        if password_protected:
            password = self.get_password()
            if not password:
                self.fail(self._('Password required'))
        else:
            password = ''
        m = re.search(self.RECAPTCHA_PATTERN, self.data)
        if m is None:
            self.fail(self._('ReCaptcha key not found'))
        else:
            recaptcha_key = urllib.parse.unquote(m.group(1).strip())
        self.captcha = ReCaptcha(pyfile)
        response = self.captcha.challenge(recaptcha_key, version='2invisible')
        data = self.load(action_url, ref='https://easyupload.io/', post={'type': 'download-token', 'url': self.info['pattern']['ID'], 'value': password, 'captchatoken': response, 'method': 'regular'})
        json_data = json.loads(data)
        if json_data.get('status') == True:
            self.link = json_data['download_link']
        elif password_protected:
            self.fail(self._('Wrong password'))
        else:
            self.fail(json_data['data'])