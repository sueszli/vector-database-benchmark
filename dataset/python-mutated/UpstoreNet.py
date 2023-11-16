import json
import re
from ..anticaptchas.HCaptcha import HCaptcha
from ..base.simple_downloader import SimpleDownloader

class UpstoreNet(SimpleDownloader):
    __name__ = 'UpstoreNet'
    __type__ = 'downloader'
    __version__ = '0.18'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?(?:upstore\\.net|upsto\\.re)/(?P<ID>\\w+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Upstore.Net File Download Hoster'
    __license__ = 'GPLv3'
    __authors__ = [('igel', 'igelkun@myopera.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    INFO_PATTERN = '<div class="comment">.*?</div>\\s*\\n<h2 style="margin:0">(?P<N>.*?)</h2>\\s*\\n<div class="comment">\\s*\\n\\s*(?P<S>[\\d.,]+) (?P<U>[\\w^_]+)'
    OFFLINE_PATTERN = '<span class="error">File (?:not found|was deleted).*</span>'
    PREMIUM_ONLY_PATTERN = 'available only for Premium'
    LINK_FREE_PATTERN = '<a href="(https?://.*?)" target="_blank"><b>'
    URL_REPLACEMENTS = [(__pattern__ + '.*', 'https://upstore.net/\\g<ID>')]
    DL_LIMIT_PATTERN = 'Please wait .+? before downloading next'
    WAIT_PATTERN = 'var sec = (\\d+)'
    COOKIES = [('upstore.net', 'lang', 'en')]

    def handle_free(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        self.data = self.load(pyfile.url, post={'hash': self.info['pattern']['ID'], 'free': 'Slow download'})
        m = re.search(self.WAIT_PATTERN, self.data)
        if m is None:
            self.error(self._('Wait pattern not found'))
        wait_time = int(m.group(1))
        self.set_wait(wait_time)
        hcaptcha = HCaptcha(self.pyfile)
        captcha_key = hcaptcha.detect_key()
        if captcha_key is None:
            self.fail(self._('captcha key not found'))
        self.captcha = hcaptcha
        post_data = {'hash': self.info['pattern']['ID'], 'free': 'Get download link', 'antispam': 'spam', 'kpw': 'spam'}
        post_data['h-captcha-response'] = post_data['g-recaptcha-response'] = hcaptcha.challenge(captcha_key)
        self.wait()
        self.data = self.load(pyfile.url, post=post_data, ref=pyfile.url)
        if 'Captcha check failed' in self.data:
            self.captcha.invalid()
        else:
            self.captcha.correct()
        self.check_errors()
        m = re.search(self.LINK_FREE_PATTERN, self.data)
        if m is not None:
            self.link = m.group(1)

    def handle_premium(self, pyfile):
        if False:
            i = 10
            return i + 15
        self.data = self.load('https://upstore.net/load/premium', post={'hash': self.info['pattern']['ID'], 'antispam': 'spam', 'js': '1'})
        json_data = json.loads(self.data)
        self.link = json_data['ok']