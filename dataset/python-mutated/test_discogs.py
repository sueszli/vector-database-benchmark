"""Tests for discogs plugin.
"""
import unittest
from test import _common
from test._common import Bag
from test.helper import capture_log
from beets import config
from beets.util.id_extractors import extract_discogs_id_regex
from beetsplug.discogs import DiscogsPlugin

class DGAlbumInfoTest(_common.TestCase):

    def _make_release(self, tracks=None):
        if False:
            while True:
                i = 10
        'Returns a Bag that mimics a discogs_client.Release. The list\n        of elements on the returned Bag is incomplete, including just\n        those required for the tests on this class.'
        data = {'id': 'ALBUM ID', 'uri': 'https://www.discogs.com/release/release/13633721', 'title': 'ALBUM TITLE', 'year': '3001', 'artists': [{'name': 'ARTIST NAME', 'id': 'ARTIST ID', 'join': ','}], 'formats': [{'descriptions': ['FORMAT DESC 1', 'FORMAT DESC 2'], 'name': 'FORMAT', 'qty': 1}], 'styles': ['STYLE1', 'STYLE2'], 'genres': ['GENRE1', 'GENRE2'], 'labels': [{'name': 'LABEL NAME', 'catno': 'CATALOG NUMBER'}], 'tracklist': []}
        if tracks:
            for recording in tracks:
                data['tracklist'].append(recording)
        return Bag(data=data, title=data['title'], artists=[Bag(data=d) for d in data['artists']])

    def _make_track(self, title, position='', duration='', type_=None):
        if False:
            for i in range(10):
                print('nop')
        track = {'title': title, 'position': position, 'duration': duration}
        if type_ is not None:
            track['type_'] = type_
        return track

    def _make_release_from_positions(self, positions):
        if False:
            while True:
                i = 10
        'Return a Bag that mimics a discogs_client.Release with a\n        tracklist where tracks have the specified `positions`.'
        tracks = [self._make_track('TITLE%s' % i, position) for (i, position) in enumerate(positions, start=1)]
        return self._make_release(tracks)

    def test_parse_media_for_tracks(self):
        if False:
            while True:
                i = 10
        tracks = [self._make_track('TITLE ONE', '1', '01:01'), self._make_track('TITLE TWO', '2', '02:02')]
        release = self._make_release(tracks=tracks)
        d = DiscogsPlugin().get_album_info(release)
        t = d.tracks
        self.assertEqual(d.media, 'FORMAT')
        self.assertEqual(t[0].media, d.media)
        self.assertEqual(t[1].media, d.media)

    def test_parse_medium_numbers_single_medium(self):
        if False:
            print('Hello World!')
        release = self._make_release_from_positions(['1', '2'])
        d = DiscogsPlugin().get_album_info(release)
        t = d.tracks
        self.assertEqual(d.mediums, 1)
        self.assertEqual(t[0].medium, 1)
        self.assertEqual(t[0].medium_total, 2)
        self.assertEqual(t[1].medium, 1)
        self.assertEqual(t[0].medium_total, 2)

    def test_parse_medium_numbers_two_mediums(self):
        if False:
            i = 10
            return i + 15
        release = self._make_release_from_positions(['1-1', '2-1'])
        d = DiscogsPlugin().get_album_info(release)
        t = d.tracks
        self.assertEqual(d.mediums, 2)
        self.assertEqual(t[0].medium, 1)
        self.assertEqual(t[0].medium_total, 1)
        self.assertEqual(t[1].medium, 2)
        self.assertEqual(t[1].medium_total, 1)

    def test_parse_medium_numbers_two_mediums_two_sided(self):
        if False:
            for i in range(10):
                print('nop')
        release = self._make_release_from_positions(['A1', 'B1', 'C1'])
        d = DiscogsPlugin().get_album_info(release)
        t = d.tracks
        self.assertEqual(d.mediums, 2)
        self.assertEqual(t[0].medium, 1)
        self.assertEqual(t[0].medium_total, 2)
        self.assertEqual(t[0].medium_index, 1)
        self.assertEqual(t[1].medium, 1)
        self.assertEqual(t[1].medium_total, 2)
        self.assertEqual(t[1].medium_index, 2)
        self.assertEqual(t[2].medium, 2)
        self.assertEqual(t[2].medium_total, 1)
        self.assertEqual(t[2].medium_index, 1)

    def test_parse_track_indices(self):
        if False:
            for i in range(10):
                print('nop')
        release = self._make_release_from_positions(['1', '2'])
        d = DiscogsPlugin().get_album_info(release)
        t = d.tracks
        self.assertEqual(t[0].medium_index, 1)
        self.assertEqual(t[0].index, 1)
        self.assertEqual(t[0].medium_total, 2)
        self.assertEqual(t[1].medium_index, 2)
        self.assertEqual(t[1].index, 2)
        self.assertEqual(t[1].medium_total, 2)

    def test_parse_track_indices_several_media(self):
        if False:
            return 10
        release = self._make_release_from_positions(['1-1', '1-2', '2-1', '3-1'])
        d = DiscogsPlugin().get_album_info(release)
        t = d.tracks
        self.assertEqual(d.mediums, 3)
        self.assertEqual(t[0].medium_index, 1)
        self.assertEqual(t[0].index, 1)
        self.assertEqual(t[0].medium_total, 2)
        self.assertEqual(t[1].medium_index, 2)
        self.assertEqual(t[1].index, 2)
        self.assertEqual(t[1].medium_total, 2)
        self.assertEqual(t[2].medium_index, 1)
        self.assertEqual(t[2].index, 3)
        self.assertEqual(t[2].medium_total, 1)
        self.assertEqual(t[3].medium_index, 1)
        self.assertEqual(t[3].index, 4)
        self.assertEqual(t[3].medium_total, 1)

    def test_parse_position(self):
        if False:
            i = 10
            return i + 15
        'Test the conversion of discogs `position` to medium, medium_index\n        and subtrack_index.'
        positions = [('1', (None, '1', None)), ('A12', ('A', '12', None)), ('12-34', ('12-', '34', None)), ('CD1-1', ('CD1-', '1', None)), ('1.12', (None, '1', '12')), ('12.a', (None, '12', 'A')), ('12.34', (None, '12', '34')), ('1ab', (None, '1', 'AB')), ('IV', ('IV', None, None))]
        d = DiscogsPlugin()
        for (position, expected) in positions:
            self.assertEqual(d.get_track_index(position), expected)

    def test_parse_tracklist_without_sides(self):
        if False:
            i = 10
            return i + 15
        'Test standard Discogs position 12.2.9#1: "without sides".'
        release = self._make_release_from_positions(['1', '2', '3'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 3)

    def test_parse_tracklist_with_sides(self):
        if False:
            while True:
                i = 10
        'Test standard Discogs position 12.2.9#2: "with sides".'
        release = self._make_release_from_positions(['A1', 'A2', 'B1', 'B2'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 4)

    def test_parse_tracklist_multiple_lp(self):
        if False:
            i = 10
            return i + 15
        'Test standard Discogs position 12.2.9#3: "multiple LP".'
        release = self._make_release_from_positions(['A1', 'A2', 'B1', 'C1'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 2)
        self.assertEqual(len(d.tracks), 4)

    def test_parse_tracklist_multiple_cd(self):
        if False:
            for i in range(10):
                print('nop')
        'Test standard Discogs position 12.2.9#4: "multiple CDs".'
        release = self._make_release_from_positions(['1-1', '1-2', '2-1', '3-1'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 3)
        self.assertEqual(len(d.tracks), 4)

    def test_parse_tracklist_non_standard(self):
        if False:
            i = 10
            return i + 15
        'Test non standard Discogs position.'
        release = self._make_release_from_positions(['I', 'II', 'III', 'IV'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 4)

    def test_parse_tracklist_subtracks_dot(self):
        if False:
            for i in range(10):
                print('nop')
        'Test standard Discogs position 12.2.9#5: "sub tracks, dots".'
        release = self._make_release_from_positions(['1', '2.1', '2.2', '3'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 3)
        release = self._make_release_from_positions(['A1', 'A2.1', 'A2.2', 'A3'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 3)

    def test_parse_tracklist_subtracks_letter(self):
        if False:
            while True:
                i = 10
        'Test standard Discogs position 12.2.9#5: "sub tracks, letter".'
        release = self._make_release_from_positions(['A1', 'A2a', 'A2b', 'A3'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 3)
        release = self._make_release_from_positions(['A1', 'A2.a', 'A2.b', 'A3'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 3)

    def test_parse_tracklist_subtracks_extra_material(self):
        if False:
            print('Hello World!')
        'Test standard Discogs position 12.2.9#6: "extra material".'
        release = self._make_release_from_positions(['1', '2', 'Video 1'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 2)
        self.assertEqual(len(d.tracks), 3)

    def test_parse_tracklist_subtracks_indices(self):
        if False:
            i = 10
            return i + 15
        'Test parsing of subtracks that include index tracks.'
        release = self._make_release_from_positions(['', '', '1.1', '1.2'])
        release.data['tracklist'][0]['title'] = 'MEDIUM TITLE'
        release.data['tracklist'][1]['title'] = 'TRACK GROUP TITLE'
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(d.tracks[0].disctitle, 'MEDIUM TITLE')
        self.assertEqual(len(d.tracks), 1)
        self.assertEqual(d.tracks[0].title, 'TRACK GROUP TITLE')

    def test_parse_tracklist_subtracks_nested_logical(self):
        if False:
            i = 10
            return i + 15
        'Test parsing of subtracks defined inside a index track that are\n        logical subtracks (ie. should be grouped together into a single track).\n        '
        release = self._make_release_from_positions(['1', '', '3'])
        release.data['tracklist'][1]['title'] = 'TRACK GROUP TITLE'
        release.data['tracklist'][1]['sub_tracks'] = [self._make_track('TITLE ONE', '2.1', '01:01'), self._make_track('TITLE TWO', '2.2', '02:02')]
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 3)
        self.assertEqual(d.tracks[1].title, 'TRACK GROUP TITLE')

    def test_parse_tracklist_subtracks_nested_physical(self):
        if False:
            while True:
                i = 10
        'Test parsing of subtracks defined inside a index track that are\n        physical subtracks (ie. should not be grouped together).\n        '
        release = self._make_release_from_positions(['1', '', '4'])
        release.data['tracklist'][1]['title'] = 'TRACK GROUP TITLE'
        release.data['tracklist'][1]['sub_tracks'] = [self._make_track('TITLE ONE', '2', '01:01'), self._make_track('TITLE TWO', '3', '02:02')]
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 1)
        self.assertEqual(len(d.tracks), 4)
        self.assertEqual(d.tracks[1].title, 'TITLE ONE')
        self.assertEqual(d.tracks[2].title, 'TITLE TWO')

    def test_parse_tracklist_disctitles(self):
        if False:
            i = 10
            return i + 15
        'Test parsing of index tracks that act as disc titles.'
        release = self._make_release_from_positions(['', '1-1', '1-2', '', '2-1'])
        release.data['tracklist'][0]['title'] = 'MEDIUM TITLE CD1'
        release.data['tracklist'][3]['title'] = 'MEDIUM TITLE CD2'
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.mediums, 2)
        self.assertEqual(d.tracks[0].disctitle, 'MEDIUM TITLE CD1')
        self.assertEqual(d.tracks[1].disctitle, 'MEDIUM TITLE CD1')
        self.assertEqual(d.tracks[2].disctitle, 'MEDIUM TITLE CD2')
        self.assertEqual(len(d.tracks), 3)

    def test_parse_minimal_release(self):
        if False:
            return 10
        'Test parsing of a release with the minimal amount of information.'
        data = {'id': 123, 'uri': 'https://www.discogs.com/release/123456-something', 'tracklist': [self._make_track('A', '1', '01:01')], 'artists': [{'name': 'ARTIST NAME', 'id': 321, 'join': ''}], 'title': 'TITLE'}
        release = Bag(data=data, title=data['title'], artists=[Bag(data=d) for d in data['artists']])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.artist, 'ARTIST NAME')
        self.assertEqual(d.album, 'TITLE')
        self.assertEqual(len(d.tracks), 1)

    def test_parse_release_without_required_fields(self):
        if False:
            return 10
        'Test parsing of a release that does not have the required fields.'
        release = Bag(data={}, refresh=lambda *args: None)
        with capture_log() as logs:
            d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d, None)
        self.assertIn('Release does not contain the required fields', logs[0])

    def test_album_for_id(self):
        if False:
            i = 10
            return i + 15
        'Test parsing for a valid Discogs release_id'
        test_patterns = [('http://www.discogs.com/G%C3%BCnther-Lause-Meru-Ep/release/4354798', 4354798), ('http://www.discogs.com/release/4354798-G%C3%BCnther-Lause-Meru-Ep', 4354798), ('http://www.discogs.com/G%C3%BCnther-4354798Lause-Meru-Ep/release/4354798', 4354798), ('http://www.discogs.com/release/4354798-G%C3%BCnther-4354798Lause-Meru-Ep/', 4354798), ('[r4354798]', 4354798), ('r4354798', 4354798), ('4354798', 4354798), ('yet-another-metadata-provider.org/foo/12345', ''), ('005b84a0-ecd6-39f1-b2f6-6eb48756b268', '')]
        for (test_pattern, expected) in test_patterns:
            match = extract_discogs_id_regex(test_pattern)
            if not match:
                match = ''
            self.assertEqual(match, expected)

    def test_default_genre_style_settings(self):
        if False:
            print('Hello World!')
        'Test genre default settings, genres to genre, styles to style'
        release = self._make_release_from_positions(['1', '2'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.genre, 'GENRE1, GENRE2')
        self.assertEqual(d.style, 'STYLE1, STYLE2')

    def test_append_style_to_genre(self):
        if False:
            while True:
                i = 10
        'Test appending style to genre if config enabled'
        config['discogs']['append_style_genre'] = True
        release = self._make_release_from_positions(['1', '2'])
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.genre, 'GENRE1, GENRE2, STYLE1, STYLE2')
        self.assertEqual(d.style, 'STYLE1, STYLE2')

    def test_append_style_to_genre_no_style(self):
        if False:
            return 10
        'Test nothing appended to genre if style is empty'
        config['discogs']['append_style_genre'] = True
        release = self._make_release_from_positions(['1', '2'])
        release.data['styles'] = []
        d = DiscogsPlugin().get_album_info(release)
        self.assertEqual(d.genre, 'GENRE1, GENRE2')
        self.assertEqual(d.style, None)

def suite():
    if False:
        return 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')