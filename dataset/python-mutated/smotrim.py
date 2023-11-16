from .common import InfoExtractor
from ..utils import ExtractorError

class SmotrimIE(InfoExtractor):
    _VALID_URL = 'https?://smotrim\\.ru/(?P<type>brand|video|article|live)/(?P<id>[0-9]+)'
    _TESTS = [{'url': 'https://smotrim.ru/video/1539617', 'md5': 'b1923a533c8cab09679789d720d0b1c5', 'info_dict': {'id': '1539617', 'ext': 'mp4', 'title': 'Полиглот. Китайский с нуля за 16 часов! Урок №16', 'description': ''}, 'add_ie': ['RUTV']}, {'url': 'https://smotrim.ru/article/2813445', 'md5': 'e0ac453952afbc6a2742e850b4dc8e77', 'info_dict': {'id': '2431846', 'ext': 'mp4', 'title': 'Новости культуры. Съёмки первой программы "Большие и маленькие"', 'description': 'md5:94a4a22472da4252bf5587a4ee441b99'}, 'add_ie': ['RUTV']}, {'url': 'https://smotrim.ru/brand/64356', 'md5': '740472999ccff81d7f6df79cecd91c18', 'info_dict': {'id': '2354523', 'ext': 'mp4', 'title': 'Большие и маленькие. Лучшее. 4-й выпуск', 'description': 'md5:84089e834429008371ea41ea3507b989'}, 'add_ie': ['RUTV']}, {'url': 'https://smotrim.ru/live/19201', 'info_dict': {'id': '19201', 'ext': 'mp4', 'title': 'Россия Культура. Прямой эфир', 'description': ''}, 'add_ie': ['RUTV']}]

    def _real_extract(self, url):
        if False:
            for i in range(10):
                print('nop')
        (video_id, typ) = self._match_valid_url(url).group('id', 'type')
        rutv_type = 'video'
        if typ not in ('video', 'live'):
            webpage = self._download_webpage(url, video_id, f'Resolving {typ} link')
            video_id = self._search_regex('"https://player.smotrim.ru/iframe/video/id/(?P<video_id>\\d+)/', webpage, 'video_id', default=None)
            if not video_id:
                raise ExtractorError('There are no video in this page.', expected=True)
        elif typ == 'live':
            rutv_type = 'live'
        return self.url_result(f'https://player.vgtrk.com/iframe/{rutv_type}/id/{video_id}')