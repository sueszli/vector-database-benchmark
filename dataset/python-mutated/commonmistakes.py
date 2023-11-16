from .common import InfoExtractor
from ..utils import ExtractorError

class CommonMistakesIE(InfoExtractor):
    IE_DESC = False
    _VALID_URL = '(?:url|URL|yt-dlp)$'
    _TESTS = [{'url': 'url', 'only_matching': True}, {'url': 'URL', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            return 10
        msg = 'You\'ve asked yt-dlp to download the URL "%s". That doesn\'t make any sense. Simply remove the parameter in your command or configuration.' % url
        if not self.get_param('verbose'):
            msg += ' Add -v to the command line to see what arguments and configuration yt-dlp has'
        raise ExtractorError(msg, expected=True)

class UnicodeBOMIE(InfoExtractor):
    IE_DESC = False
    _VALID_URL = '(?P<bom>\\ufeff)(?P<id>.*)$'
    _TESTS = [{'url': '\ufeffhttp://www.youtube.com/watch?v=BaW_jenozKc', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            while True:
                i = 10
        real_url = self._match_id(url)
        self.report_warning('Your URL starts with a Byte Order Mark (BOM). Removing the BOM and looking for "%s" ...' % real_url)
        return self.url_result(real_url)