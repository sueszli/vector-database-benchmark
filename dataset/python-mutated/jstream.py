import base64
import re
import json
from .common import InfoExtractor
from ..utils import float_or_none, js_to_json, remove_start

class JStreamIE(InfoExtractor):
    _VALID_URL = 'jstream:(?P<host>www\\d+):(?P<id>(?P<publisher>[a-z0-9]+):(?P<mid>\\d+))'
    _TESTS = [{'url': 'jstream:www50:eqd638pvwx:752', 'info_dict': {'id': 'eqd638pvwx:752', 'ext': 'mp4', 'title': '阪神淡路大震災 激震の記録2020年版\u3000解説動画', 'duration': 672, 'thumbnail': 're:https?://eqd638pvwx\\.eq\\.webcdn\\.stream\\.ne\\.jp/.+\\.jpg'}}]

    def _parse_jsonp(self, callback, string, video_id):
        if False:
            return 10
        return self._search_json(f'\\s*{re.escape(callback)}\\s*\\(', string, callback, video_id)

    def _find_formats(self, video_id, movie_list_hls, host, publisher, subtitles):
        if False:
            return 10
        for value in movie_list_hls:
            text = value.get('text') or ''
            if not text.startswith('auto'):
                continue
            m3u8_id = remove_start(remove_start(text, 'auto'), '_') or None
            (fmts, subs) = self._extract_m3u8_formats_and_subtitles(f"https://{publisher}.eq.webcdn.stream.ne.jp/{host}/{publisher}/jmc_pub/{value.get('url')}", video_id, 'mp4', m3u8_id=m3u8_id)
            self._merge_subtitles(subs, target=subtitles)
            yield from fmts

    def _real_extract(self, url):
        if False:
            print('Hello World!')
        (host, publisher, mid, video_id) = self._match_valid_url(url).group('host', 'publisher', 'mid', 'id')
        video_info_jsonp = self._download_webpage(f'https://{publisher}.eq.webcdn.stream.ne.jp/{host}/{publisher}/jmc_pub/eq_meta/v1/{mid}.jsonp', video_id, 'Requesting video info')
        video_info = self._parse_jsonp('metaDataResult', video_info_jsonp, video_id)['movie']
        subtitles = {}
        formats = list(self._find_formats(video_id, video_info.get('movie_list_hls'), host, publisher, subtitles))
        self._remove_duplicate_formats(formats)
        return {'id': video_id, 'title': video_info.get('title'), 'duration': float_or_none(video_info.get('duration')), 'thumbnail': video_info.get('thumbnail_url'), 'formats': formats, 'subtitles': subtitles}

    @classmethod
    def _extract_embed_urls(cls, url, webpage):
        if False:
            for i in range(10):
                print('nop')
        script_tag = re.search('<script\\s*[^>]+?src="https://ssl-cache\\.stream\\.ne\\.jp/(?P<host>www\\d+)/(?P<publisher>[a-z0-9]+)/[^"]+?/if\\.js"', webpage)
        if not script_tag:
            return
        (host, publisher) = script_tag.groups()
        for m in re.finditer('(?s)PlayerFactoryIF\\.create\\(\\s*({[^\\}]+?})\\s*\\)\\s*;', webpage):
            info = json.loads(js_to_json(m.group(1)))
            mid = base64.b64decode(info.get('m')).decode()
            yield f'jstream:{host}:{publisher}:{mid}'