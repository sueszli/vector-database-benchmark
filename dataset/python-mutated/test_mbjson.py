from test.picardtestcase import PicardTestCase, load_test_json
from picard import config
from picard.album import Album
from picard.const import ALIAS_TYPE_ARTIST_NAME_ID, ALIAS_TYPE_LEGAL_NAME_ID, ALIAS_TYPE_SEARCH_HINT_ID
from picard.mbjson import _locales_from_aliases, _node_skip_empty_iter, _parse_attributes, _relations_to_metadata_target_type_url, _translate_artist_node, artist_to_metadata, countries_from_node, get_score, label_info_from_node, media_formats_from_node, medium_to_metadata, recording_to_metadata, release_dates_and_countries_from_node, release_group_to_metadata, release_to_metadata, track_to_metadata
from picard.metadata import Metadata
from picard.releasegroup import ReleaseGroup
from picard.track import Track
settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': False, 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en']}

class MBJSONItersTest(PicardTestCase):

    def test_node_skip_empty_iter(self):
        if False:
            i = 10
            return i + 15
        d = {'bool_false': False, 'bool_true': True, 'int_0': 0, 'int_1': 1, 'float_0': 0.0, 'float_1': 1.1, 'list_empty': [], 'list_non_empty': ['a'], 'dict_empty': {}, 'dict_non_empty': {'a': 'b'}}
        expected = set(d) - {'list_empty', 'dict_empty'}
        result = set({k: v for (k, v) in _node_skip_empty_iter(d)})
        self.assertSetEqual(expected, result)

class MBJSONTest(PicardTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.init_test(self.filename)

    def init_test(self, filename):
        if False:
            return 10
        self.set_config_values(settings)
        self.json_doc = load_test_json(filename)

class ReleaseTest(MBJSONTest):
    filename = 'release.json'

    def test_release(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        a = Album('1')
        release_to_metadata(self.json_doc, m, a)
        self.assertEqual(m['album'], 'The Dark Side of the Moon')
        self.assertEqual(m['albumartist'], 'Pink Floyd')
        self.assertEqual(m['albumartistsort'], 'Pink Floyd')
        self.assertEqual(m['asin'], 'b123')
        self.assertEqual(m['barcode'], '123')
        self.assertEqual(m['catalognumber'], 'SHVL 804')
        self.assertEqual(m['date'], '1973-03-24')
        self.assertEqual(m['label'], 'Harvest')
        self.assertEqual(m['musicbrainz_albumartistid'], '83d91898-7763-47d7-b03b-b92132375c47')
        self.assertEqual(m['musicbrainz_albumid'], 'b84ee12a-09ef-421b-82de-0441a926375b')
        self.assertEqual(m['producer'], 'Hipgnosis')
        self.assertEqual(m['releasecountry'], 'GB')
        self.assertEqual(m['releasestatus'], 'official')
        self.assertEqual(m['script'], 'Latn')
        self.assertEqual(m['~albumartists'], 'Pink Floyd')
        self.assertEqual(m['~albumartists_sort'], 'Pink Floyd')
        self.assertEqual(m['~releasecomment'], 'stereo')
        self.assertEqual(m['~releaseannotation'], 'Original Vinyl release')
        self.assertEqual(m['~releaselanguage'], 'eng')
        self.assertEqual(m.getall('~releasecountries'), ['GB', 'NZ'])
        self.assertEqual(m['~release_series'], 'Why Pink Floyd?')
        self.assertEqual(m['~release_seriesid'], '7421b602-a413-4151-bcf4-d831debc3f27')
        self.assertEqual(m['~release_seriescomment'], 'Pink Floyed special editions')
        self.assertEqual(m['~release_seriesnumber'], '')
        self.assertEqual(a.genres, {'genre1': 6, 'genre2': 3, 'tag1': 6, 'tag2': 3})
        for artist in a._album_artists:
            self.assertEqual(artist.genres, {'british': 2, 'progressive rock': 10})

    def test_release_without_release_relationships(self):
        if False:
            i = 10
            return i + 15
        config.setting['release_ars'] = False
        m = Metadata()
        a = Album('1')
        release_to_metadata(self.json_doc, m, a)
        self.assertEqual(m['album'], 'The Dark Side of the Moon')
        self.assertEqual(m['albumartist'], 'Pink Floyd')
        self.assertEqual(m['albumartistsort'], 'Pink Floyd')
        self.assertEqual(m['asin'], 'b123')
        self.assertEqual(m['barcode'], '123')
        self.assertEqual(m['catalognumber'], 'SHVL 804')
        self.assertEqual(m['date'], '1973-03-24')
        self.assertEqual(m['label'], 'Harvest')
        self.assertEqual(m['musicbrainz_albumartistid'], '83d91898-7763-47d7-b03b-b92132375c47')
        self.assertEqual(m['musicbrainz_albumid'], 'b84ee12a-09ef-421b-82de-0441a926375b')
        self.assertEqual(m['producer'], '')
        self.assertEqual(m['releasecountry'], 'GB')
        self.assertEqual(m['releasestatus'], 'official')
        self.assertEqual(m['script'], 'Latn')
        self.assertEqual(m['~albumartists'], 'Pink Floyd')
        self.assertEqual(m['~albumartists_sort'], 'Pink Floyd')
        self.assertEqual(m['~releaselanguage'], 'eng')
        self.assertEqual(m.getall('~releasecountries'), ['GB', 'NZ'])
        self.assertEqual(a.genres, {'genre1': 6, 'genre2': 3, 'tag1': 6, 'tag2': 3})
        for artist in a._album_artists:
            self.assertEqual(artist.genres, {'british': 2, 'progressive rock': 10})

    def test_preferred_release_country(self):
        if False:
            for i in range(10):
                print('nop')
        m = Metadata()
        a = Album('1')
        release_to_metadata(self.json_doc, m, a)
        self.assertEqual(m['releasecountry'], 'GB')
        config.setting['preferred_release_countries'] = ['NZ', 'GB']
        release_to_metadata(self.json_doc, m, a)
        self.assertEqual(m['releasecountry'], 'NZ')
        config.setting['preferred_release_countries'] = ['GB', 'NZ']
        release_to_metadata(self.json_doc, m, a)
        self.assertEqual(m['releasecountry'], 'GB')

    def test_media_formats_from_node(self):
        if False:
            print('Hello World!')
        formats = media_formats_from_node(self.json_doc['media'])
        self.assertEqual(formats, '12" Vinyl')

    def test_release_group_rels(self):
        if False:
            print('Hello World!')
        m = Metadata()
        release_group_to_metadata(self.json_doc['release-group'], m)
        self.assertEqual(m.getall('~releasegroup_series'), ["Absolute Radio's The 100 Collection", '1001 Albums You Must Hear Before You Die'])
        self.assertEqual(m.getall('~releasegroup_seriesid'), ['4bf41050-6fa9-41a6-8398-15bdab4b0352', '4bc2a338-e1d8-4546-8a61-640da8aaf888'])
        self.assertEqual(m.getall('~releasegroup_seriescomment'), ['2005 edition'])
        self.assertEqual(m.getall('~releasegroup_seriesnumber'), ['15', '291'])

    def test_release_group_rels_double(self):
        if False:
            return 10
        m = Metadata()
        release_group_to_metadata(self.json_doc['release-group'], m)
        release_group_to_metadata(self.json_doc['release-group'], m)
        self.assertEqual(m.getall('~releasegroup_series'), ["Absolute Radio's The 100 Collection", '1001 Albums You Must Hear Before You Die'])
        self.assertEqual(m.getall('~releasegroup_seriesid'), ['4bf41050-6fa9-41a6-8398-15bdab4b0352', '4bc2a338-e1d8-4546-8a61-640da8aaf888'])
        self.assertEqual(m.getall('~releasegroup_seriescomment'), ['2005 edition'])
        self.assertEqual(m.getall('~releasegroup_seriesnumber'), ['15', '291'])

    def test_release_group_rels_removed(self):
        if False:
            print('Hello World!')
        m = Metadata()
        release_group_to_metadata(self.json_doc['release-group'], m)
        for (i, rel) in enumerate(self.json_doc['release-group']['relations']):
            if not rel['type'] == 'part of':
                continue
            if rel['series']['name'] == '1001 Albums You Must Hear Before You Die':
                del self.json_doc['release-group']['relations'][i]
                break
        release_group_to_metadata(self.json_doc['release-group'], m)
        self.assertEqual(m.getall('~releasegroup_series'), ["Absolute Radio's The 100 Collection"])
        self.assertEqual(m.getall('~releasegroup_seriesid'), ['4bf41050-6fa9-41a6-8398-15bdab4b0352'])
        self.assertEqual(m.getall('~releasegroup_seriescomment'), [])
        self.assertEqual(m.getall('~releasegroup_seriesnumber'), ['15'])

class NullReleaseTest(MBJSONTest):
    filename = 'release_null.json'

    def test_release(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        a = Album('1')
        release_to_metadata(self.json_doc, m, a)
        self.assertEqual(m, {})

    def test_media_formats_from_node(self):
        if False:
            print('Hello World!')
        formats = media_formats_from_node(self.json_doc['media'])
        self.assertEqual(formats, '(unknown)')

class RecordingTest(MBJSONTest):
    filename = 'recording.json'

    def test_recording(self):
        if False:
            for i in range(10):
                print('nop')
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['artist'], 'Ed Sheeran')
        self.assertEqual(m['artists'], 'Ed Sheeran')
        self.assertEqual(m['artistsort'], 'Sheeran, Ed')
        self.assertEqual(m['isrc'], 'GBAHS1400099')
        self.assertEqual(m['language'], 'eng')
        self.assertEqual(m['musicbrainz_artistid'], 'b8a7c51f-362c-4dcb-a259-bc6e0095f0a6')
        self.assertEqual(m['musicbrainz_recordingid'], 'cb2cc207-8125-445c-9ef9-6ea44eee959a')
        self.assertEqual(m['musicbrainz_workid'], 'dc469dc8-198e-42e5-b5a7-6be2f0a95ac0')
        self.assertEqual(m['performer:'], 'Ed Sheeran')
        self.assertEqual(m['performer:lead vocals'], 'Ed Sheeran')
        self.assertEqual(m['performer:guitar family'], 'Ed Sheeran')
        self.assertEqual(m['title'], 'Thinking Out Loud')
        self.assertEqual(m['work'], 'Thinking Out Loud')
        self.assertEqual(m['~workcomment'], 'Ed Sheeran song')
        self.assertEqual(m['writer'], 'Ed Sheeran; Amy Wadge')
        self.assertEqual(m['~writersort'], 'Sheeran, Ed; Wadge, Amy')
        self.assertEqual(m['~artists_sort'], 'Sheeran, Ed')
        self.assertEqual(m['~length'], '4:41')
        self.assertEqual(m['~recordingtitle'], 'Thinking Out Loud')
        self.assertEqual(m['~recording_firstreleasedate'], '2014-06-20')
        self.assertEqual(m['~video'], '')
        self.assertNotIn('originaldate', m)
        self.assertNotIn('originalyear', m)
        self.assertEqual(t.genres, {'blue-eyed soul': 1, 'pop': 3})
        for artist in t._track_artists:
            self.assertEqual(artist.genres, {'dance-pop': 1, 'guitarist': 0})

    def test_recording_instrument_credits(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        t = Track('1')
        config.setting['standardize_instruments'] = False
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['performer:vocals'], 'Ed Sheeran')
        self.assertEqual(m['performer:acoustic guitar'], 'Ed Sheeran')

class RecordingComposerCreditsTest(MBJSONTest):
    filename = 'recording_composer.json'

    def test_standardize_artists(self):
        if False:
            print('Hello World!')
        m = Metadata()
        t = Track('1')
        config.setting['translate_artist_names'] = False
        config.setting['standardize_artists'] = True
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['composer'], 'Пётр Ильич Чайковский')
        self.assertEqual(m['composersort'], 'Tchaikovsky, Pyotr Ilyich')

    def test_use_credited_as(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        t = Track('1')
        config.setting['translate_artist_names'] = False
        config.setting['standardize_artists'] = False
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['composer'], 'Tchaikovsky')
        self.assertEqual(m['composersort'], 'Tchaikovsky, Pyotr Ilyich')

    def test_translate(self):
        if False:
            for i in range(10):
                print('nop')
        m = Metadata()
        t = Track('1')
        config.setting['translate_artist_names'] = True
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['composer'], 'Pyotr Ilyich Tchaikovsky')
        self.assertEqual(m['composersort'], 'Tchaikovsky, Pyotr Ilyich')

class RecordingInstrumentalTest(MBJSONTest):
    filename = 'recording_instrumental.json'

    def test_recording(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        self.assertIn('instrumental', m.getall('~performance_attributes'))
        self.assertEqual(m['language'], 'zxx')
        self.assertNotIn('lyricist', m)

class MultiWorkRecordingTest(MBJSONTest):
    filename = 'recording_multiple_works.json'

    def test_recording(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        self.assertIn('instrumental', m.getall('~performance_attributes'))
        self.assertEqual(m['language'], 'jpn; eng; zxx')
        self.assertEqual(m['lyricist'], 'Satoru Kōsaki; Aki Hata; Minoru Shiraishi')
        self.assertEqual(m['~lyricistsort'], 'Kōsaki, Satoru; Hata, Aki; Shiraishi, Minoru')

class RecordingVideoTest(MBJSONTest):
    filename = 'recording_video.json'

    def test_recording(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['director'], 'Edward 209')
        self.assertEqual(m['producer'], 'Edward 209')
        self.assertEqual(m['~video'], '1')

class NullRecordingTest(MBJSONTest):
    filename = 'recording_null.json'

    def test_recording(self):
        if False:
            print('Hello World!')
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m, {})

class RecordingCreditsTest(MBJSONTest):
    filename = 'recording_credits.json'

    def test_recording_solo_vocals(self):
        if False:
            for i in range(10):
                print('nop')
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        config.setting['standardize_artists'] = False
        self.assertNotIn('performer:solo', m)
        self.assertEqual(m['performer:solo vocals'], 'Frida')

    def test_recording_standardize_artist_credits(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        t = Track('1')
        config.setting['standardize_artists'] = True
        recording_to_metadata(self.json_doc, m, t)
        self.assertNotIn('performer:solo', m)
        self.assertEqual(m['performer:solo vocals'], 'Anni-Frid Lyngstad')

    def test_recording_instrument_keep_case(self):
        if False:
            return 10
        m = Metadata()
        t = Track('1')
        recording_to_metadata(self.json_doc, m, t)
        self.assertEqual(m['performer:EWI'], 'Michael Brecker')

class TrackTest(MBJSONTest):
    filename = 'track.json'

    def test_track(self):
        if False:
            while True:
                i = 10
        t = Track('1')
        m = t.metadata
        track_to_metadata(self.json_doc, t)
        self.assertEqual(m['title'], 'Speak to Me')
        self.assertEqual(m['musicbrainz_recordingid'], 'bef3fddb-5aca-49f5-b2fd-d56a23268d63')
        self.assertEqual(m['musicbrainz_trackid'], 'd4156411-b884-368f-a4cb-7c0101a557a2')
        self.assertEqual(m['~length'], '1:08')
        self.assertEqual(m['tracknumber'], '1')
        self.assertEqual(m['~musicbrainz_tracknumber'], 'A1')
        self.assertEqual(m['~recordingcomment'], 'original stereo mix')
        self.assertEqual(m['~recordingtitle'], 'Speak to Me')

class PregapTrackTest(MBJSONTest):
    filename = 'track_pregap.json'

    def test_track(self):
        if False:
            for i in range(10):
                print('nop')
        t = Track('1')
        m = t.metadata
        track_to_metadata(self.json_doc, t)
        self.assertEqual(m['title'], 'Lady')
        self.assertEqual(m['tracknumber'], '0')
        self.assertEqual(m['~musicbrainz_tracknumber'], '0')

class NullTrackTest(MBJSONTest):
    filename = 'track_null.json'

    def test_track(self):
        if False:
            while True:
                i = 10
        t = Track('1')
        m = t.metadata
        track_to_metadata(self.json_doc, t)
        self.assertEqual(m, {})

class MediaTest(MBJSONTest):
    filename = 'release_5medias.json'

    def test_media_formats_from_node_multi(self):
        if False:
            i = 10
            return i + 15
        formats = media_formats_from_node(self.json_doc['media'])
        self.assertEqual('2×CD + 2×DVD-Video + Blu-ray', formats)

    def test_medium_to_metadata_0(self):
        if False:
            print('Hello World!')
        m = Metadata()
        medium_to_metadata(self.json_doc['media'][0], m)
        self.assertEqual(m['discnumber'], '1')
        self.assertEqual(m['media'], 'CD')
        self.assertEqual(m['totaltracks'], '5')
        self.assertEqual(m['discsubtitle'], 'The Original Album')

    def test_medium_to_metadata_4(self):
        if False:
            for i in range(10):
                print('nop')
        m = Metadata()
        medium_to_metadata(self.json_doc['media'][4], m)
        self.assertEqual(m['discnumber'], '5')
        self.assertEqual(m['media'], 'Blu-ray')
        self.assertEqual(m['totaltracks'], '19')
        self.assertEqual(m['discsubtitle'], 'High Resolution Audio and Audio‐Visual Material')

class MediaPregapTest(MBJSONTest):
    filename = 'media_pregap.json'

    def test_track(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        medium_to_metadata(self.json_doc, m)
        self.assertEqual(m['discnumber'], '1')
        self.assertEqual(m['media'], 'Enhanced CD')
        self.assertEqual(m['totaltracks'], '9')

class NullMediaTest(MBJSONTest):
    filename = 'media_null.json'

    def test_track(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        medium_to_metadata(self.json_doc, m)
        self.assertEqual(m, {})

class NullArtistTest(MBJSONTest):
    filename = 'artist_null.json'

    def test_artist(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        artist_to_metadata(self.json_doc, m)
        self.assertEqual(m, {})

class ArtistEndedTest(MBJSONTest):
    filename = 'artist_ended.json'

    def test_artist_ended(self):
        if False:
            return 10
        m = Metadata()
        artist_to_metadata(self.json_doc, m)
        self.assertEqual(m['area'], 'France')
        self.assertEqual(m['beginarea'], 'Paris')
        self.assertEqual(m['begindate'], '1928-04-02')
        self.assertEqual(m['endarea'], 'Paris')
        self.assertEqual(m['enddate'], '1991-03-02')
        self.assertEqual(m['gender'], 'Male')
        self.assertEqual(m['musicbrainz_artistid'], 'b21ef19b-c6aa-4775-90d3-3cc3e067ce6d')
        self.assertEqual(m['name'], 'Serge Gainsbourg')
        self.assertEqual(m['type'], 'Person')

class ArtistTranslationTest(MBJSONTest):
    filename = 'artist.json'

    def test_locale_specific_match_first(self):
        if False:
            print('Hello World!')
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': False, 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en_CA', 'en']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'Ed Sheeran (en_CA)')

    def test_locale_specific_match_first_exc(self):
        if False:
            print('Hello World!')
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': True, 'script_exceptions': [('LATIN', 0)], 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en_CA', 'en']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'Ed Sheeran')

    def test_locale_specific_match_second(self):
        if False:
            i = 10
            return i + 15
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': False, 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en_UK', 'en']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'Ed Sheeran (en)')

    def test_artist_match_root_locale_fallback(self):
        if False:
            while True:
                i = 10
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': False, 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en_UK']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'Ed Sheeran (en)')

    def test_artist_no_match(self):
        if False:
            return 10
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': False, 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['de']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'Ed Sheeran')

class ArtistTranslationArabicExceptionsTest(MBJSONTest):
    filename = 'artist_arabic.json'

    def test_locale_specific_match_first_exc1(self):
        if False:
            i = 10
            return i + 15
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': True, 'script_exceptions': [('LATIN', 0)], 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en_CA', 'en']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'Mohamed Mounir')

    def test_locale_specific_match_first_exc2(self):
        if False:
            for i in range(10):
                print('nop')
        settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': True, 'translate_artist_names_script_exception': True, 'script_exceptions': [('ARABIC', 0)], 'standardize_instruments': True, 'release_ars': True, 'preferred_release_countries': [], 'artist_locales': ['en_CA', 'en']}
        self.set_config_values(settings)
        (artist_name, artist_sort_name) = _translate_artist_node(self.json_doc)
        self.assertEqual(artist_name, 'محمد منير')

class TestAliasesLocales(PicardTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.maxDiff = None
        self.aliases = [{'name': 'Shearan', 'sort-name': 'Shearan', 'primary': None, 'locale': None, 'type-id': ALIAS_TYPE_SEARCH_HINT_ID}, {'primary': True, 'name': 'Ed Sheeran (en)', 'sort-name': 'Sheeran, Ed', 'type-id': ALIAS_TYPE_ARTIST_NAME_ID, 'locale': 'en'}, {'primary': True, 'name': 'Ed Sheeran (en_CA)', 'sort-name': 'Sheeran, Ed', 'type-id': ALIAS_TYPE_ARTIST_NAME_ID, 'locale': 'en_CA'}]

    def test_1(self):
        if False:
            return 10
        expect_full = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed')), 'en_CA': (0.8, ('Ed Sheeran (en_CA)', 'Sheeran, Ed'))}
        expect_root = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed'))}
        (full_locales, root_locales) = _locales_from_aliases(self.aliases)
        self.assertDictEqual(expect_full, full_locales)
        self.assertDictEqual(expect_root, root_locales)

    def test_2(self):
        if False:
            while True:
                i = 10
        self.aliases[2]['type-id'] = ALIAS_TYPE_LEGAL_NAME_ID
        expect_full = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed')), 'en_CA': (0.65, ('Ed Sheeran (en_CA)', 'Sheeran, Ed'))}
        expect_root = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed'))}
        (full_locales, root_locales) = _locales_from_aliases(self.aliases)
        self.assertDictEqual(expect_full, full_locales)
        self.assertDictEqual(expect_root, root_locales)

    def test_3(self):
        if False:
            print('Hello World!')
        self.aliases[0]['primary'] = True
        del self.aliases[0]['locale']
        expect_full = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed')), 'en_CA': (0.8, ('Ed Sheeran (en_CA)', 'Sheeran, Ed'))}
        expect_root = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed'))}
        (full_locales, root_locales) = _locales_from_aliases(self.aliases)
        self.assertDictEqual(expect_full, full_locales)
        self.assertDictEqual(expect_root, root_locales)

    def test_4(self):
        if False:
            return 10
        self.aliases[2]['type-id'] = ALIAS_TYPE_SEARCH_HINT_ID
        expect_full = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed')), 'en_CA': (0.4, ('Ed Sheeran (en_CA)', 'Sheeran, Ed'))}
        expect_root = {'en': (0.8, ('Ed Sheeran (en)', 'Sheeran, Ed'))}
        (full_locales, root_locales) = _locales_from_aliases(self.aliases)
        self.assertDictEqual(expect_full, full_locales)
        self.assertDictEqual(expect_root, root_locales)

    def test_5(self):
        if False:
            print('Hello World!')
        self.aliases[1]['locale'] = 'en_US'
        self.aliases[1]['name'] = 'Ed Sheeran (en_US)'
        expect_full = {'en_US': (0.8, ('Ed Sheeran (en_US)', 'Sheeran, Ed')), 'en_CA': (0.8, ('Ed Sheeran (en_CA)', 'Sheeran, Ed'))}
        expect_root = {'en': (0.6, ('Ed Sheeran (en_US)', 'Sheeran, Ed'))}
        (full_locales, root_locales) = _locales_from_aliases(self.aliases)
        self.assertDictEqual(expect_full, full_locales)
        self.assertDictEqual(expect_root, root_locales)

    def test_6(self):
        if False:
            i = 10
            return i + 15
        self.aliases[2]['locale'] = 'en'
        self.aliases[2]['name'] = 'Ed Sheeran (en2)'
        self.aliases[2]['type-id'] = ALIAS_TYPE_ARTIST_NAME_ID
        self.aliases[1]['type-id'] = ALIAS_TYPE_LEGAL_NAME_ID
        self.aliases[1]['name'] = 'Ed Sheeran (en1)'
        expect_full = {'en': (0.8, ('Ed Sheeran (en2)', 'Sheeran, Ed'))}
        expect_root = {'en': (0.8, ('Ed Sheeran (en2)', 'Sheeran, Ed'))}
        (full_locales, root_locales) = _locales_from_aliases(self.aliases)
        self.assertDictEqual(expect_full, full_locales)
        self.assertDictEqual(expect_root, root_locales)

class ReleaseGroupTest(MBJSONTest):
    filename = 'release_group.json'

    def test_release_group(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        r = ReleaseGroup('1')
        release_group_to_metadata(self.json_doc, m, r)
        self.assertEqual(m['musicbrainz_releasegroupid'], 'f5093c06-23e3-404f-aeaa-40f72885ee3a')
        self.assertEqual(m['~releasegroup_firstreleasedate'], '1973-03-24')
        self.assertEqual(m['originaldate'], '1973-03-24')
        self.assertEqual(m['originalyear'], '1973')
        self.assertEqual(m['releasetype'], 'album')
        self.assertEqual(m['~primaryreleasetype'], 'album')
        self.assertEqual(m['~releasegroup'], 'The Dark Side of the Moon')
        self.assertEqual(r.genres, {'test2': 3, 'test': 6})

class NullReleaseGroupTest(MBJSONTest):
    filename = 'release_group_null.json'

    def test_release_group(self):
        if False:
            print('Hello World!')
        m = Metadata()
        r = ReleaseGroup('1')
        release_group_to_metadata(self.json_doc, m, r)
        self.assertEqual(m, {})

class CountriesFromNodeTest(MBJSONTest):
    filename = 'country.json'

    def test_countries_from_node(self):
        if False:
            return 10
        countries = countries_from_node(self.json_doc)
        self.assertEqual(['GB'], countries)

    def test_countries_from_node_no_event(self):
        if False:
            return 10
        del self.json_doc['release-events']
        countries = countries_from_node(self.json_doc)
        self.assertEqual([], countries)

    def test_countries_from_node_no_area(self):
        if False:
            for i in range(10):
                print('nop')
        del self.json_doc['release-events'][0]['area']
        countries = countries_from_node(self.json_doc)
        self.assertEqual([], countries)

class CountriesFromNodeNullTest(MBJSONTest):
    filename = 'country_null.json'

    def test_countries_from_node(self):
        if False:
            return 10
        countries = countries_from_node(self.json_doc)
        self.assertEqual(countries, [])

class DatesCountriesFromNodeTest(MBJSONTest):
    filename = 'country.json'

    def test_dates_countries_from_node(self):
        if False:
            while True:
                i = 10
        (dates, countries) = release_dates_and_countries_from_node(self.json_doc)
        self.assertEqual(['GB'], countries)
        self.assertEqual(['1986-03'], dates)

    def test_dates_countries_from_node_no_event(self):
        if False:
            for i in range(10):
                print('nop')
        del self.json_doc['release-events']
        (dates, countries) = release_dates_and_countries_from_node(self.json_doc)
        self.assertEqual([], countries)
        self.assertEqual([], dates)

class DatesCountriesFromNodeNullTest(MBJSONTest):
    filename = 'country_null.json'

    def test_dates_countries_from_node(self):
        if False:
            i = 10
            return i + 15
        (dates, countries) = release_dates_and_countries_from_node(self.json_doc)
        self.assertEqual(countries, [])
        self.assertEqual([''], dates)

class LabelInfoTest(MBJSONTest):
    filename = 'label_info.json'

    def _label_info(self, n):
        if False:
            for i in range(10):
                print('nop')
        return label_info_from_node(self.json_doc['releases'][n]['label-info'])

    def test_label_info_from_node_0(self):
        if False:
            return 10
        self.assertEqual((['naïve'], ['NJ628311']), self._label_info(0))

    def test_label_info_from_node_1(self):
        if False:
            print('Hello World!')
        self.assertEqual((['naïve'], []), self._label_info(1))

    def test_label_info_from_node_2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual((['naïve'], []), self._label_info(2))

    def test_label_info_from_node_3(self):
        if False:
            while True:
                i = 10
        self.assertEqual(([], ['[None]']), self._label_info(3))

class NullLabelInfoTest(MBJSONTest):
    filename = 'label_info_null.json'

    def test_label_info_from_node_0(self):
        if False:
            i = 10
            return i + 15
        label_info = label_info_from_node(self.json_doc['releases'][0]['label-info'])
        self.assertEqual(label_info, ([], []))

class GetScoreTest(PicardTestCase):

    def test_get_score(self):
        if False:
            for i in range(10):
                print('nop')
        for (score, expected) in ((42, 0.42), ('100', 1.0), (0, 0.0), (None, 1.0), ('', 1.0)):
            self.assertEqual(expected, get_score({'score': score}))

    def test_get_score_no_score(self):
        if False:
            return 10
        self.assertEqual(1.0, get_score({}))

class ParseAttributeTest(PicardTestCase):

    def test_1(self):
        if False:
            i = 10
            return i + 15
        (attrs, reltype, attr_credits) = (('guest', 'keyboard'), 'instrument', {'keyboard': 'keyboards'})
        result = _parse_attributes(attrs, reltype, attr_credits)
        expected = 'guest keyboards'
        self.assertEqual(expected, result)

    def test_2(self):
        if False:
            i = 10
            return i + 15
        (attrs, reltype, attr_credits) = ((), 'vocal', {})
        result = _parse_attributes(attrs, reltype, attr_credits)
        expected = 'vocals'
        self.assertEqual(expected, result)

    def test_3(self):
        if False:
            for i in range(10):
                print('nop')
        (attrs, reltype, attr_credits) = (('guitar', 'keyboard'), 'instrument', {'keyboard': 'keyboards', 'guitar': 'weird guitar'})
        result = _parse_attributes(attrs, reltype, attr_credits)
        expected = 'weird guitar and keyboards'
        self.assertEqual(expected, result)

class RelationsToMetadataTargetTypeUrlTest(PicardTestCase):

    def test_invalid_asin_url(self):
        if False:
            return 10
        m = Metadata()
        relation = {'type': 'amazon asin', 'url': {'resource': 'http://www.amazon.com/dp/020530902x'}}
        _relations_to_metadata_target_type_url(relation, m, None)
        self.assertEqual('', m['asin'])

    def test_has_asin_already(self):
        if False:
            while True:
                i = 10
        m = Metadata({'asin': 'ASIN'})
        relation = {'type': 'amazon asin', 'url': {'resource': 'http://www.amazon.com/dp/020530902X'}}
        _relations_to_metadata_target_type_url(relation, m, None)
        self.assertEqual('ASIN', m['asin'])

    def test_valid_asin_url(self):
        if False:
            while True:
                i = 10
        m = Metadata()
        relation = {'type': 'amazon asin', 'url': {'resource': 'http://www.amazon.com/dp/020530902X'}}
        _relations_to_metadata_target_type_url(relation, m, None)
        self.assertEqual('020530902X', m['asin'])

    def test_license_url(self):
        if False:
            i = 10
            return i + 15
        m = Metadata()
        relation = {'type': 'license', 'url': {'resource': 'https://URL.LICENSE'}}
        _relations_to_metadata_target_type_url(relation, m, None)
        self.assertEqual('https://URL.LICENSE', m['license'])