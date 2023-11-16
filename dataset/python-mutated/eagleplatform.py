import functools
import re
from .common import InfoExtractor
from ..networking.exceptions import HTTPError
from ..utils import ExtractorError, int_or_none, smuggle_url, unsmuggle_url, url_or_none

class EaglePlatformIE(InfoExtractor):
    _VALID_URL = '(?x)\n                    (?:\n                        eagleplatform:(?P<custom_host>[^/]+):|\n                        https?://(?P<host>.+?\\.media\\.eagleplatform\\.com)/index/player\\?.*\\brecord_id=\n                    )\n                    (?P<id>\\d+)\n                '
    _EMBED_REGEX = ['<iframe[^>]+src=(["\\\'])(?P<url>(?:https?:)?//.+?\\.media\\.eagleplatform\\.com/index/player\\?.+?)\\1']
    _TESTS = [{'url': 'http://lentaru.media.eagleplatform.com/index/player?player=new&record_id=227304&player_template_id=5201', 'info_dict': {'id': '227304', 'ext': 'mp4', 'title': 'Навальный вышел на свободу', 'description': 'md5:d97861ac9ae77377f3f20eaf9d04b4f5', 'thumbnail': 're:^https?://.*\\.jpg$', 'duration': 87, 'view_count': int, 'age_limit': 0}}, {'url': 'eagleplatform:media.clipyou.ru:12820', 'md5': '358597369cf8ba56675c1df15e7af624', 'info_dict': {'id': '12820', 'ext': 'mp4', 'title': "'O Sole Mio", 'thumbnail': 're:^https?://.*\\.jpg$', 'duration': 216, 'view_count': int}, 'skip': 'Georestricted'}, {'url': 'eagleplatform:tvrainru.media.eagleplatform.com:582306', 'only_matching': True}]

    @classmethod
    def _extract_embed_urls(cls, url, webpage):
        if False:
            i = 10
            return i + 15
        add_referer = functools.partial(smuggle_url, data={'referrer': url})
        res = tuple(super()._extract_embed_urls(url, webpage))
        if res:
            return map(add_referer, res)
        PLAYER_JS_RE = '\n                        <script[^>]+\n                            src=(?P<qjs>["\\\'])(?:https?:)?//(?P<host>(?:(?!(?P=qjs)).)+\\.media\\.eagleplatform\\.com)/player/player\\.js(?P=qjs)\n                        .+?\n                    '
        mobj = re.search('(?xs)\n                    %s\n                    <div[^>]+\n                        class=(?P<qclass>["\\\'])eagleplayer(?P=qclass)[^>]+\n                        data-id=["\\\'](?P<id>\\d+)\n            ' % PLAYER_JS_RE, webpage)
        if mobj is not None:
            return [add_referer('eagleplatform:%(host)s:%(id)s' % mobj.groupdict())]
        mobj = re.search('(?xs)\n                    %s\n                    <script>\n                    .+?\n                    new\\s+EaglePlayer\\(\n                        (?:[^,]+\\s*,\\s*)?\n                        {\n                            .+?\n                            \\bid\\s*:\\s*["\\\']?(?P<id>\\d+)\n                            .+?\n                        }\n                    \\s*\\)\n                    .+?\n                    </script>\n            ' % PLAYER_JS_RE, webpage)
        if mobj is not None:
            return [add_referer('eagleplatform:%(host)s:%(id)s' % mobj.groupdict())]

    @staticmethod
    def _handle_error(response):
        if False:
            i = 10
            return i + 15
        status = int_or_none(response.get('status', 200))
        if status != 200:
            raise ExtractorError(' '.join(response['errors']), expected=True)

    def _download_json(self, url_or_request, video_id, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            response = super(EaglePlatformIE, self)._download_json(url_or_request, video_id, *args, **kwargs)
        except ExtractorError as ee:
            if isinstance(ee.cause, HTTPError):
                response = self._parse_json(ee.cause.response.read().decode('utf-8'), video_id)
                self._handle_error(response)
            raise
        return response

    def _get_video_url(self, url_or_request, video_id, note='Downloading JSON metadata'):
        if False:
            print('Hello World!')
        return self._download_json(url_or_request, video_id, note)['data'][0]

    def _real_extract(self, url):
        if False:
            print('Hello World!')
        (url, smuggled_data) = unsmuggle_url(url, {})
        mobj = self._match_valid_url(url)
        (host, video_id) = (mobj.group('custom_host') or mobj.group('host'), mobj.group('id'))
        headers = {}
        query = {'id': video_id}
        referrer = smuggled_data.get('referrer')
        if referrer:
            headers['Referer'] = referrer
            query['referrer'] = referrer
        player_data = self._download_json('http://%s/api/player_data' % host, video_id, headers=headers, query=query)
        media = player_data['data']['playlist']['viewports'][0]['medialist'][0]
        title = media['title']
        description = media.get('description')
        thumbnail = self._proto_relative_url(media.get('snapshot'), 'http:')
        duration = int_or_none(media.get('duration'))
        view_count = int_or_none(media.get('views'))
        age_restriction = media.get('age_restriction')
        age_limit = None
        if age_restriction:
            age_limit = 0 if age_restriction == 'allow_all' else 18
        secure_m3u8 = self._proto_relative_url(media['sources']['secure_m3u8']['auto'], 'http:')
        formats = []
        m3u8_url = self._get_video_url(secure_m3u8, video_id, 'Downloading m3u8 JSON')
        m3u8_formats = self._extract_m3u8_formats(m3u8_url, video_id, 'mp4', entry_protocol='m3u8_native', m3u8_id='hls', fatal=False)
        formats.extend(m3u8_formats)
        m3u8_formats_dict = {}
        for f in m3u8_formats:
            if f.get('height') is not None:
                m3u8_formats_dict[f['height']] = f
        mp4_data = self._download_json(re.sub('m3u8|hlsvod|hls|f4m', 'mp4s', secure_m3u8), video_id, 'Downloading mp4 JSON', fatal=False)
        if mp4_data:
            for (format_id, format_url) in mp4_data.get('data', {}).items():
                if not url_or_none(format_url):
                    continue
                height = int_or_none(format_id)
                if height is not None and m3u8_formats_dict.get(height):
                    f = m3u8_formats_dict[height].copy()
                    f.update({'format_id': f['format_id'].replace('hls', 'http'), 'protocol': 'http'})
                else:
                    f = {'format_id': 'http-%s' % format_id, 'height': int_or_none(format_id)}
                f['url'] = format_url
                formats.append(f)
        return {'id': video_id, 'title': title, 'description': description, 'thumbnail': thumbnail, 'duration': duration, 'view_count': view_count, 'age_limit': age_limit, 'formats': formats}

class ClipYouEmbedIE(InfoExtractor):
    _VALID_URL = False

    @classmethod
    def _extract_embed_urls(cls, url, webpage):
        if False:
            for i in range(10):
                print('nop')
        mobj = re.search('<iframe[^>]+src="https?://(?P<host>media\\.clipyou\\.ru)/index/player\\?.*\\brecord_id=(?P<id>\\d+).*"', webpage)
        if mobj is not None:
            yield smuggle_url('eagleplatform:%(host)s:%(id)s' % mobj.groupdict(), {'referrer': url})