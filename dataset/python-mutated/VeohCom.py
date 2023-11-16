import json
from ..base.simple_downloader import SimpleDownloader

class VeohCom(SimpleDownloader):
    __name__ = 'VeohCom'
    __type__ = 'downloader'
    __version__ = '0.30'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?veoh\\.com/(?:tv/)?(?:watch|videos)/(?P<ID>v\\w+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Veoh.com downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    NAME_PATTERN = '<meta name="title" content="(?P<N>.*?)"'
    OFFLINE_PATTERN = ">Sorry, we couldn\\'t find the video you were looking for"
    URL_REPLACEMENTS = [(__pattern__ + '.*', 'https://www.veoh.com/watch/\\g<ID>')]
    COOKIES = [('veoh.com', 'lassieLocale', 'en')]

    def setup(self):
        if False:
            while True:
                i = 10
        self.resume_download = True
        self.multi_dl = True
        self.chunk_limit = -1

    def handle_free(self, pyfile):
        if False:
            i = 10
            return i + 15
        video_id = self.info['pattern']['ID']
        video_data = json.loads(self.load(f'https://www.veoh.com/watch/getVideo/{video_id}'))
        pyfile.name = video_data['video']['title'] + '.mp4'
        self.link = video_data['video']['src']['HQ']