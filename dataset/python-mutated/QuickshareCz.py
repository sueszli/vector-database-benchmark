import re
from datetime import timedelta
from ..base.simple_downloader import SimpleDownloader

class QuickshareCz(SimpleDownloader):
    __name__ = 'QuickshareCz'
    __type__ = 'downloader'
    __version__ = '0.64'
    __status__ = 'testing'
    __pattern__ = 'http://(?:[^/]*\\.)?quickshare\\.cz/stahnout-soubor/.+'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Quickshare.cz downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('zoidberg', 'zoidberg@mujmail.cz')]
    NAME_PATTERN = '<th width="145px">Název:</th>\\s*<td style="word-wrap:break-word;">(?P<N>.+?)</td>'
    SIZE_PATTERN = '<th>Velikost:</th>\\s*<td>(?P<S>[\\d.,]+) (?P<U>[\\w^_]+)</td>'
    OFFLINE_PATTERN = '<script type="text/javascript">location\\.href=\\\'/chyba\\\';</script>'

    def process(self, pyfile):
        if False:
            return 10
        self.data = self.load(pyfile.url)
        self.get_file_info()
        self.jsvars = {x: y.strip("'") for (x, y) in re.findall("var (\\w+) = ([\\d.]+|'.+?')", self.data)}
        self.log_debug(self.jsvars)
        pyfile.name = self.jsvars['ID3']
        if self.premium:
            if 'UU_prihlasen' in self.jsvars:
                if self.jsvars['UU_prihlasen'] == '0':
                    self.log_warning(self._('User not logged in'))
                    self.account.relogin()
                    self.retry()
                elif float(self.jsvars['UU_kredit']) < float(self.jsvars['kredit_odecet']):
                    self.log_warning(self._('Not enough credit left'))
                    self.premium = False
        if self.premium:
            self.handle_premium(pyfile)
        else:
            self.handle_free(pyfile)
        if self.scan_download({'error': re.compile(b'\\AChyba!')}, read_size=100):
            self.fail(self._('File not found or plugin defect'))

    def handle_free(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        download_url = '{}/download.php'.format(self.jsvars['server'])
        data = {x: self.jsvars[x] for x in self.jsvars if x in ('ID1', 'ID2', 'ID3', 'ID4')}
        self.log_debug('FREE URL1:' + download_url, data)
        header = self.load(download_url, post=data, just_header=True)
        self.link = header.get('location')
        if not self.link:
            self.fail(self._('File not found'))
        self.log_debug('FREE URL2:' + self.link)
        m = re.search('/chyba/(\\d+)', self.link)
        if m is not None:
            if m.group(1) == '1':
                self.retry(60, timedelta(minutes=2).total_seconds(), 'This IP is already downloading')
            elif m.group(1) == '2':
                self.retry(60, 60, 'No free slots available')
            else:
                self.fail(self._('Error {}').format(m.group(1)))

    def handle_premium(self, pyfile):
        if False:
            while True:
                i = 10
        download_url = '{}/download_premium.php'.format(self.jsvars['server'])
        data = {x: self.jsvars[x] for x in self.jsvars if x in ('ID1', 'ID2', 'ID4', 'ID5')}
        self.download(download_url, get=data)