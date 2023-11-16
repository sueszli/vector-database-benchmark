import os
import re
from ..anticaptchas.ReCaptcha import ReCaptcha
from ..base.simple_downloader import SimpleDownloader

class FilerNet(SimpleDownloader):
    __name__ = 'FilerNet'
    __type__ = 'downloader'
    __version__ = '0.31'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?filer\\.net/get/\\w+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Filer.net downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('stickell', 'l.stickell@yahoo.it'), ('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    INFO_PATTERN = '<h1 class="page-header">Free Download (?P<N>\\S+) <small>(?P<S>[\\w.]+) (?P<U>[\\w^_]+)</small></h1>'
    OFFLINE_PATTERN = 'Datei +nicht mehr vorhanden'
    TEMP_OFFLINE_PATTERN = 'Leider sind alle kostenlosen Download-Slots belegt'
    WAIT_PATTERN = 'var count = (\\d+);'
    LINK_PATTERN = 'href="([^"]+)">Get download</a>'

    def handle_free(self, pyfile):
        if False:
            while True:
                i = 10
        inputs = self.parse_html_form(input_names={'token': re.compile('.+')})[1]
        if inputs is None or 'token' not in inputs:
            self.retry()
        self.data = self.load(pyfile.url, post={'token': inputs['token']})
        inputs = self.parse_html_form(input_names={'hash': re.compile('.+')})[1]
        if inputs is None or 'hash' not in inputs:
            self.error(self._('Unable to detect hash'))
        self.captcha = ReCaptcha(pyfile)
        response = self.captcha.challenge()
        self.download(pyfile.url, post={'g-recaptcha-response': response, 'hash': inputs['hash']})
        if self.scan_download({'html': re.compile(b'\\A\\s*<!DOCTYPE html')}) == 'html':
            with open(self.last_download, 'r') as f:
                self.data = f.read()
            os.remove(self.last_download)
            if re.search(self.TEMP_OFFLINE_PATTERN, self.data) is not None:
                self.temp_offline()
            else:
                return SimpleDownloader.check_download(self)
        else:
            return SimpleDownloader.check_download(self)