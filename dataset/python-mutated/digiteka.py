from .common import InfoExtractor
from ..utils import int_or_none

class DigitekaIE(InfoExtractor):
    _VALID_URL = '(?x)\n        https?://(?:www\\.)?(?:digiteka\\.net|ultimedia\\.com)/\n        (?:\n            deliver/\n            (?P<embed_type>\n                generic|\n                musique\n            )\n            (?:/[^/]+)*/\n            (?:\n                src|\n                article\n            )|\n            default/index/video\n            (?P<site_type>\n                generic|\n                music\n            )\n            /id\n        )/(?P<id>[\\d+a-z]+)'
    _EMBED_REGEX = ['<(?:iframe|script)[^>]+src=["\\\'](?P<url>(?:https?:)?//(?:www\\.)?ultimedia\\.com/deliver/(?:generic|musique)(?:/[^/]+)*/(?:src|article)/[\\d+a-z]+)']
    _TESTS = [{'url': 'https://www.ultimedia.com/default/index/videogeneric/id/s8uk0r', 'md5': '276a0e49de58c7e85d32b057837952a2', 'info_dict': {'id': 's8uk0r', 'ext': 'mp4', 'title': 'Loi sur la fin de vie: le texte prévoit un renforcement des directives anticipées', 'thumbnail': 're:^https?://.*\\.jpg', 'duration': 74, 'upload_date': '20150317', 'timestamp': 1426604939, 'uploader_id': '3fszv'}}, {'url': 'https://www.ultimedia.com/default/index/videomusic/id/xvpfp8', 'md5': '2ea3513813cf230605c7e2ffe7eca61c', 'info_dict': {'id': 'xvpfp8', 'ext': 'mp4', 'title': "Two - C'est La Vie (clip)", 'thumbnail': 're:^https?://.*\\.jpg', 'duration': 233, 'upload_date': '20150224', 'timestamp': 1424760500, 'uploader_id': '3rfzk'}}, {'url': 'https://www.digiteka.net/deliver/generic/iframe/mdtk/01637594/src/lqm3kl/zone/1/showtitle/1/autoplay/yes', 'only_matching': True}]

    def _real_extract(self, url):
        if False:
            while True:
                i = 10
        mobj = self._match_valid_url(url)
        video_id = mobj.group('id')
        video_type = mobj.group('embed_type') or mobj.group('site_type')
        if video_type == 'music':
            video_type = 'musique'
        deliver_info = self._download_json('http://www.ultimedia.com/deliver/video?video=%s&topic=%s' % (video_id, video_type), video_id)
        yt_id = deliver_info.get('yt_id')
        if yt_id:
            return self.url_result(yt_id, 'Youtube')
        jwconf = deliver_info['jwconf']
        formats = []
        for source in jwconf['playlist'][0]['sources']:
            formats.append({'url': source['file'], 'format_id': source.get('label')})
        title = deliver_info['title']
        thumbnail = jwconf.get('image')
        duration = int_or_none(deliver_info.get('duration'))
        timestamp = int_or_none(deliver_info.get('release_time'))
        uploader_id = deliver_info.get('owner_id')
        return {'id': video_id, 'title': title, 'thumbnail': thumbnail, 'duration': duration, 'timestamp': timestamp, 'uploader_id': uploader_id, 'formats': formats}