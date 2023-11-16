import re
from pyload.core.utils.misc import eval_js
from ..base.simple_downloader import SimpleDownloader

class SmuleCom(SimpleDownloader):
    __name__ = 'SmuleCom'
    __type__ = 'downloader'
    __version__ = '0.05'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.)?smule\\.com/recording/.+'
    __description__ = 'SmuleCom downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('igel', None), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    MEDIA_URL_PATTERN = 'initPlayer\\(.+?["\\\']video_media_url["\\\']:["\\\'](.+?)["\\\']'
    JS_HEADER_PATTERN = '(?P<decoder>function \\w+\\(\\w+\\){.+?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\\+.+?}).+?;(?P<initvars>var r=.+?;)'
    JS_PROCESS_PATTERN = 'processRecording\\s*=\\s*function.+?}'
    JS_SPLIT_WORD = 'EXIF'
    NAME_PATTERN = 'initPlayer\\(.+?["\\\']title["\\\']:["\\\'](?P<N>.+?)["\\\']'
    COMMUNITY_JS_PATTERN = '<script.+?src=["\\\']/*(\\w[^"\\\']*community.+?js)["\\\']'
    OFFLINE_PATTERN = 'Page Not Found'

    def get_info(self, url='', html=''):
        if False:
            print('Hello World!')
        info = super(SimpleDownloader, self).get_info(url, html)
        if 'name' in info:
            info['name'] += '.mp4'
        return info

    def handle_free(self, pyfile):
        if False:
            for i in range(10):
                print('nop')
        m = re.search(self.MEDIA_URL_PATTERN, self.data)
        if m is None:
            self.fail(self._('Could not find any media URLs'))
        encoded_media_url = m.group(1)
        self.log_debug(f'Found encoded media URL: {encoded_media_url}')
        m = re.search(self.COMMUNITY_JS_PATTERN, self.data)
        if m is None:
            self.fail(self._('Could not find necessary javascript script to load'))
        community_js_url = m.group(1)
        self.log_debug(f'Found community js at {community_js_url}')
        community_js_code = self.load(community_js_url)
        community_js_code = community_js_code.partition(self.JS_SPLIT_WORD)[0]
        m = re.search(self.JS_HEADER_PATTERN, community_js_code)
        if m is None:
            self.fail(self._('Could not parse the necessary parts off the javascript'))
        decoder_function = m.group('decoder')
        initialization = m.group('initvars')
        m = re.search(self.JS_PROCESS_PATTERN, community_js_code)
        if m is None:
            self.fail(self._('Could not parse the processing function off the javascript'))
        process_function = m.group(0)
        new_js_code = decoder_function + '; ' + initialization + '; var ' + process_function + '; processRecording("' + encoded_media_url + '");'
        self.log_debug(f'Running js script: {new_js_code}')
        js_result = eval_js(new_js_code)
        self.log_debug(f'Result is: {js_result}')
        self.link = js_result