import re
from .common import InfoExtractor
from ..utils import ExtractorError, decode_packed_codes, determine_ext, int_or_none, js_to_json, urlencode_postdata

def aa_decode(aa_code):
    if False:
        return 10
    symbol_table = [('7', '((ﾟｰﾟ) + (o^_^o))'), ('6', '((o^_^o) +(o^_^o))'), ('5', '((ﾟｰﾟ) + (ﾟΘﾟ))'), ('2', '((o^_^o) - (ﾟΘﾟ))'), ('4', '(ﾟｰﾟ)'), ('3', '(o^_^o)'), ('1', '(ﾟΘﾟ)'), ('0', '(c^_^o)')]
    delim = '(ﾟДﾟ)[ﾟεﾟ]+'
    ret = ''
    for aa_char in aa_code.split(delim):
        for (val, pat) in symbol_table:
            aa_char = aa_char.replace(pat, val)
        aa_char = aa_char.replace('+ ', '')
        m = re.match('^\\d+', aa_char)
        if m:
            ret += chr(int(m.group(0), 8))
        else:
            m = re.match('^u([\\da-f]+)', aa_char)
            if m:
                ret += chr(int(m.group(1), 16))
    return ret

class XFileShareIE(InfoExtractor):
    _SITES = (('aparat\\.cam', 'Aparat'), ('clipwatching\\.com', 'ClipWatching'), ('gounlimited\\.to', 'GoUnlimited'), ('govid\\.me', 'GoVid'), ('holavid\\.com', 'HolaVid'), ('streamty\\.com', 'Streamty'), ('thevideobee\\.to', 'TheVideoBee'), ('uqload\\.com', 'Uqload'), ('vidbom\\.com', 'VidBom'), ('vidlo\\.us', 'vidlo'), ('vidlocker\\.xyz', 'VidLocker'), ('vidshare\\.tv', 'VidShare'), ('vup\\.to', 'VUp'), ('wolfstream\\.tv', 'WolfStream'), ('xvideosharing\\.com', 'XVideoSharing'))
    IE_DESC = 'XFileShare based sites: %s' % ', '.join(list(zip(*_SITES))[1])
    _VALID_URL = 'https?://(?:www\\.)?(?P<host>%s)/(?:embed-)?(?P<id>[0-9a-zA-Z]+)' % '|'.join((site for site in list(zip(*_SITES))[0]))
    _EMBED_REGEX = ['<iframe\\b[^>]+\\bsrc=(["\\\'])(?P<url>(?:https?:)?//(?:%s)/embed-[0-9a-zA-Z]+.*?)\\1' % '|'.join((site for site in list(zip(*_SITES))[0]))]
    _FILE_NOT_FOUND_REGEXES = ('>(?:404 - )?File Not Found<', '>The file was removed by administrator<')
    _TESTS = [{'url': 'https://uqload.com/dltx1wztngdz', 'md5': '3cfbb65e4c90e93d7b37bcb65a595557', 'info_dict': {'id': 'dltx1wztngdz', 'ext': 'mp4', 'title': 'Rick Astley Never Gonna Give You mp4', 'thumbnail': 're:https://.*\\.jpg'}}, {'url': 'http://xvideosharing.com/fq65f94nd2ve', 'md5': '4181f63957e8fe90ac836fa58dc3c8a6', 'info_dict': {'id': 'fq65f94nd2ve', 'ext': 'mp4', 'title': 'sample', 'thumbnail': 're:http://.*\\.jpg'}}, {'url': 'https://aparat.cam/n4d6dh0wvlpr', 'only_matching': True}, {'url': 'https://wolfstream.tv/nthme29v9u2x', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            while True:
                i = 10
        (host, video_id) = self._match_valid_url(url).groups()
        url = 'https://%s/' % host + ('embed-%s.html' % video_id if host in ('govid.me', 'vidlo.us') else video_id)
        webpage = self._download_webpage(url, video_id)
        if any((re.search(p, webpage) for p in self._FILE_NOT_FOUND_REGEXES)):
            raise ExtractorError('Video %s does not exist' % video_id, expected=True)
        fields = self._hidden_inputs(webpage)
        if fields.get('op') == 'download1':
            countdown = int_or_none(self._search_regex('<span id="countdown_str">(?:[Ww]ait)?\\s*<span id="cxc">(\\d+)</span>\\s*(?:seconds?)?</span>', webpage, 'countdown', default=None))
            if countdown:
                self._sleep(countdown, video_id)
            webpage = self._download_webpage(url, video_id, 'Downloading video page', data=urlencode_postdata(fields), headers={'Referer': url, 'Content-type': 'application/x-www-form-urlencoded'})
        title = (self._search_regex(('style="z-index: [0-9]+;">([^<]+)</span>', '<td nowrap>([^<]+)</td>', 'h4-fine[^>]*>([^<]+)<', '>Watch (.+)[ <]', '<h2 class="video-page-head">([^<]+)</h2>', '<h2 style="[^"]*color:#403f3d[^"]*"[^>]*>([^<]+)<', 'title\\s*:\\s*"([^"]+)"'), webpage, 'title', default=None) or self._og_search_title(webpage, default=None) or video_id).strip()
        for (regex, func) in (('(eval\\(function\\(p,a,c,k,e,d\\){.+)', decode_packed_codes), ('(ﾟ.+)', aa_decode)):
            obf_code = self._search_regex(regex, webpage, 'obfuscated code', default=None)
            if obf_code:
                webpage = webpage.replace(obf_code, func(obf_code))
        formats = []
        jwplayer_data = self._search_regex(['jwplayer\\("[^"]+"\\)\\.load\\(\\[({.+?})\\]\\);', 'jwplayer\\("[^"]+"\\)\\.setup\\(({.+?})\\);'], webpage, 'jwplayer data', default=None)
        if jwplayer_data:
            jwplayer_data = self._parse_json(jwplayer_data.replace("\\'", "'"), video_id, js_to_json)
            if jwplayer_data:
                formats = self._parse_jwplayer_data(jwplayer_data, video_id, False, m3u8_id='hls', mpd_id='dash')['formats']
        if not formats:
            urls = []
            for regex in ('(?:file|src)\\s*:\\s*(["\\\'])(?P<url>http(?:(?!\\1).)+\\.(?:m3u8|mp4|flv)(?:(?!\\1).)*)\\1', 'file_link\\s*=\\s*(["\\\'])(?P<url>http(?:(?!\\1).)+)\\1', 'addVariable\\((\\\\?["\\\'])file\\1\\s*,\\s*(\\\\?["\\\'])(?P<url>http(?:(?!\\2).)+)\\2\\)', '<embed[^>]+src=(["\\\'])(?P<url>http(?:(?!\\1).)+\\.(?:m3u8|mp4|flv)(?:(?!\\1).)*)\\1'):
                for mobj in re.finditer(regex, webpage):
                    video_url = mobj.group('url')
                    if video_url not in urls:
                        urls.append(video_url)
            sources = self._search_regex('sources\\s*:\\s*(\\[(?!{)[^\\]]+\\])', webpage, 'sources', default=None)
            if sources:
                urls.extend(self._parse_json(sources, video_id))
            formats = []
            for video_url in urls:
                if determine_ext(video_url) == 'm3u8':
                    formats.extend(self._extract_m3u8_formats(video_url, video_id, 'mp4', entry_protocol='m3u8_native', m3u8_id='hls', fatal=False))
                else:
                    formats.append({'url': video_url, 'format_id': 'sd'})
        thumbnail = self._search_regex(['<video[^>]+poster="([^"]+)"', '(?:image|poster)\\s*:\\s*["\\\'](http[^"\\\']+)["\\\'],'], webpage, 'thumbnail', default=None)
        return {'id': video_id, 'title': title, 'thumbnail': thumbnail, 'formats': formats, 'http_headers': {'Referer': url}}