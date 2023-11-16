import re
import json
import os
import pycurl
from ..base.downloader import BaseDownloader

class ZDF(BaseDownloader):
    __name__ = 'ZDF Mediathek'
    __type__ = 'downloader'
    __version__ = '0.93'
    __status__ = 'testing'
    __pattern__ = 'https://(?:www\\.)?zdf\\.de/(?P<ID>[/\\w-]+)\\.html'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'ZDF.de downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = []

    def process(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        self.data = self.load(pyfile.url)
        try:
            api_token = re.search('window\\.zdfsite\\.player\\.apiToken = "([\\d\\w]+)";', self.data).group(1)
            self.req.http.c.setopt(pycurl.HTTPHEADER, ['Api-Auth: Bearer ' + api_token])
            id = re.match(self.__pattern__, pyfile.url).group('ID')
            filename = json.loads(self.load('https://api.zdf.de/content/documents/zdf/' + id + '.json', get={'profile': 'player-3'}))
            stream_list = filename['mainVideoContent']['http://zdf.de/rels/target']['streams']['default']['extId']
            streams = json.loads(self.load('https://api.zdf.de/tmd/2/ngplayer_2_4/vod/ptmd/mediathek/' + stream_list))
            download_name = streams['priorityList'][0]['formitaeten'][0]['qualities'][0]['audio']['tracks'][0]['uri']
            self.pyfile.name = os.path.basename(id) + os.path.splitext(download_name)[1]
            self.download(download_name)
        except Exception as exc:
            self.log_error(exc)