import hashlib
import json
import re
import urllib.parse
from .ffmpeg import FFmpegPostProcessor

class SponsorBlockPP(FFmpegPostProcessor):
    EXTRACTORS = {'Youtube': 'YouTube'}
    POI_CATEGORIES = {'poi_highlight': 'Highlight'}
    NON_SKIPPABLE_CATEGORIES = {**POI_CATEGORIES, 'chapter': 'Chapter'}
    CATEGORIES = {'sponsor': 'Sponsor', 'intro': 'Intermission/Intro Animation', 'outro': 'Endcards/Credits', 'selfpromo': 'Unpaid/Self Promotion', 'preview': 'Preview/Recap', 'filler': 'Filler Tangent', 'interaction': 'Interaction Reminder', 'music_offtopic': 'Non-Music Section', **NON_SKIPPABLE_CATEGORIES}

    def __init__(self, downloader, categories=None, api='https://sponsor.ajay.app'):
        if False:
            return 10
        FFmpegPostProcessor.__init__(self, downloader)
        self._categories = tuple(categories or self.CATEGORIES.keys())
        self._API_URL = api if re.match('^https?://', api) else 'https://' + api

    def run(self, info):
        if False:
            for i in range(10):
                print('nop')
        extractor = info['extractor_key']
        if extractor not in self.EXTRACTORS:
            self.to_screen(f'SponsorBlock is not supported for {extractor}')
            return ([], info)
        self.to_screen('Fetching SponsorBlock segments')
        info['sponsorblock_chapters'] = self._get_sponsor_chapters(info, info.get('duration'))
        return ([], info)

    def _get_sponsor_chapters(self, info, duration):
        if False:
            while True:
                i = 10
        segments = self._get_sponsor_segments(info['id'], self.EXTRACTORS[info['extractor_key']])

        def duration_filter(s):
            if False:
                for i in range(10):
                    print('nop')
            start_end = s['segment']
            if start_end == (0, 0):
                return False
            if start_end[0] <= 1:
                start_end[0] = 0
            if s['category'] in self.POI_CATEGORIES.keys():
                start_end[1] += 1
            if duration and duration - start_end[1] <= 1:
                start_end[1] = duration
            diff = abs(duration - s['videoDuration']) if s['videoDuration'] else 0
            return diff < 1 or (diff < 5 and diff / (start_end[1] - start_end[0]) < 0.05)
        duration_match = [s for s in segments if duration_filter(s)]
        if len(duration_match) != len(segments):
            self.report_warning('Some SponsorBlock segments are from a video of different duration, maybe from an old version of this video')

        def to_chapter(s):
            if False:
                i = 10
                return i + 15
            ((start, end), cat) = (s['segment'], s['category'])
            title = s['description'] if cat == 'chapter' else self.CATEGORIES[cat]
            return {'start_time': start, 'end_time': end, 'category': cat, 'title': title, 'type': s['actionType'], '_categories': [(cat, start, end, title)]}
        sponsor_chapters = [to_chapter(s) for s in duration_match]
        if not sponsor_chapters:
            self.to_screen('No matching segments were found in the SponsorBlock database')
        else:
            self.to_screen(f'Found {len(sponsor_chapters)} segments in the SponsorBlock database')
        return sponsor_chapters

    def _get_sponsor_segments(self, video_id, service):
        if False:
            return 10
        hash = hashlib.sha256(video_id.encode('ascii')).hexdigest()
        url = f'{self._API_URL}/api/skipSegments/{hash[:4]}?' + urllib.parse.urlencode({'service': service, 'categories': json.dumps(self._categories), 'actionTypes': json.dumps(['skip', 'poi', 'chapter'])})
        for d in self._download_json(url) or []:
            if d['videoID'] == video_id:
                return d['segments']
        return []