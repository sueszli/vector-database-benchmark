import re
import pycurl
from pyload.core.utils.misc import eval_js
from ..anticaptchas.ReCaptcha import ReCaptcha
from ..base.simple_downloader import SimpleDownloader

class HitfileNet(SimpleDownloader):
    __name__ = 'HitfileNet'
    __type__ = 'downloader'
    __version__ = '0.03'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?(?:hitfile\\.net|hil\\.to)/(?:download/free/)?(?P<ID>\\w+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Hitfile.net downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    URL_REPLACEMENTS = [(__pattern__ + '.*', 'https://hitfile.net/\\g<ID>')]
    SIZE_REPLACEMENTS = [(' ', '')]
    COOKIES = [('hitfile.net', 'user_lang', 'en')]
    NAME_PATTERN = 'You download: .*</span><span>(?P<N>.+?)</span>'
    SIZE_PATTERN = '<span class="file-size">\\((?P<S>[\\d.,]+) (?P<U>[\\w^_]+)\\)<'
    OFFLINE_PATTERN = 'File was deleted or not found'
    TEMP_OFFLINE_PATTERN = '^unmatchable$'
    DL_LIMIT_PATTERN = "<span id='timeout'>(\\d+)</span>"
    LINK_FREE_PATTERN = '(/download/redirect/[^"\\\']+)'
    LINK_PREMIUM_PATTERN = '<a href=[\\\'"](.+?/download/redirect/[^"\\\']+)'

    def handle_free(self, pyfile):
        if False:
            print('Hello World!')
        self.free_url = 'https://hitfile.net/download/free/%s' % self.info['pattern']['ID']
        self.data = self.load(self.free_url)
        m = re.search(self.DL_LIMIT_PATTERN, self.data)
        if m is not None:
            self.retry(wait=m.group(1))
        self.solve_captcha()
        m = re.search('minLimit : (.+?),', self.data)
        if m is None:
            self.fail(self._('minLimit pattern not found'))
        wait_time = eval_js(m.group(1))
        self.wait(wait_time)
        self.req.http.c.setopt(pycurl.HTTPHEADER, ['X-Requested-With: XMLHttpRequest'])
        self.data = self.load('https://hitfile.net/download/getLinkTimeout/%s' % self.info['pattern']['ID'], ref=self.free_url)
        self.req.http.c.setopt(pycurl.HTTPHEADER, ['X-Requested-With:'])
        m = re.search(self.LINK_FREE_PATTERN, self.data)
        if m is not None:
            link = 'https://hitfile.net%s' % m.group(1)
            header = self.load(link, redirect=False, just_header=True)
            self.link = header['location']

    def solve_captcha(self):
        if False:
            i = 10
            return i + 15
        (action, inputs) = self.parse_html_form("id='captcha_form'")
        if not inputs:
            self.fail(self._('Captcha form not found'))
        self.captcha = ReCaptcha(self.pyfile)
        inputs['g-recaptcha-response'] = self.captcha.challenge()
        self.captcha.correct()
        self.data = self.load(self.free_url, post=inputs)

    def handle_premium(self, pyfile):
        if False:
            i = 10
            return i + 15
        m = re.search(self.LINK_PREMIUM_PATTERN, self.data)
        if m is not None:
            self.link = m.group(1)