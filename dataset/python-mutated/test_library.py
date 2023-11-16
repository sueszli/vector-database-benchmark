"""Tests for non-query database functions of Item.
"""
import os
import os.path
import re
import shutil
import stat
import sys
import time
import unicodedata
import unittest
from test import _common
from test._common import item
from test.helper import TestHelper
from mediafile import MediaFile, UnreadableFileError
import beets.dbcore.query
import beets.library
from beets import config, plugins, util
from beets.util import bytestring_path, syspath
np = util.normpath

class LoadTest(_common.LibTestCase):

    def test_load_restores_data_from_db(self):
        if False:
            while True:
                i = 10
        original_title = self.i.title
        self.i.title = 'something'
        self.i.load()
        self.assertEqual(original_title, self.i.title)

    def test_load_clears_dirty_flags(self):
        if False:
            i = 10
            return i + 15
        self.i.artist = 'something'
        self.assertTrue('artist' in self.i._dirty)
        self.i.load()
        self.assertTrue('artist' not in self.i._dirty)

class StoreTest(_common.LibTestCase):

    def test_store_changes_database_value(self):
        if False:
            while True:
                i = 10
        self.i.year = 1987
        self.i.store()
        new_year = self.lib._connection().execute('select year from items where title="the title"').fetchone()['year']
        self.assertEqual(new_year, 1987)

    def test_store_only_writes_dirty_fields(self):
        if False:
            while True:
                i = 10
        original_genre = self.i.genre
        self.i._values_fixed['genre'] = 'beatboxing'
        self.i.store()
        new_genre = self.lib._connection().execute('select genre from items where title="the title"').fetchone()['genre']
        self.assertEqual(new_genre, original_genre)

    def test_store_clears_dirty_flags(self):
        if False:
            return 10
        self.i.composer = 'tvp'
        self.i.store()
        self.assertTrue('composer' not in self.i._dirty)

    def test_store_album_cascades_flex_deletes(self):
        if False:
            i = 10
            return i + 15
        album = _common.album()
        album.flex1 = 'Flex-1'
        self.lib.add(album)
        item = _common.item()
        item.album_id = album.id
        item.flex1 = 'Flex-1'
        self.lib.add(item)
        del album.flex1
        album.store()
        self.assertNotIn('flex1', album)
        self.assertNotIn('flex1', album.items()[0])

class AddTest(_common.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.lib = beets.library.Library(':memory:')
        self.i = item()

    def test_item_add_inserts_row(self):
        if False:
            print('Hello World!')
        self.lib.add(self.i)
        new_grouping = self.lib._connection().execute('select grouping from items where composer="the composer"').fetchone()['grouping']
        self.assertEqual(new_grouping, self.i.grouping)

    def test_library_add_path_inserts_row(self):
        if False:
            i = 10
            return i + 15
        i = beets.library.Item.from_path(os.path.join(_common.RSRC, b'full.mp3'))
        self.lib.add(i)
        new_grouping = self.lib._connection().execute('select grouping from items where composer="the composer"').fetchone()['grouping']
        self.assertEqual(new_grouping, self.i.grouping)

class RemoveTest(_common.LibTestCase):

    def test_remove_deletes_from_db(self):
        if False:
            print('Hello World!')
        self.i.remove()
        c = self.lib._connection().execute('select * from items')
        self.assertEqual(c.fetchone(), None)

class GetSetTest(_common.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.i = item()

    def test_set_changes_value(self):
        if False:
            print('Hello World!')
        self.i.bpm = 4915
        self.assertEqual(self.i.bpm, 4915)

    def test_set_sets_dirty_flag(self):
        if False:
            while True:
                i = 10
        self.i.comp = not self.i.comp
        self.assertTrue('comp' in self.i._dirty)

    def test_set_does_not_dirty_if_value_unchanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.title = self.i.title
        self.assertTrue('title' not in self.i._dirty)

    def test_invalid_field_raises_attributeerror(self):
        if False:
            print('Hello World!')
        self.assertRaises(AttributeError, getattr, self.i, 'xyzzy')

    def test_album_fallback(self):
        if False:
            for i in range(10):
                print('nop')
        lib = beets.library.Library(':memory:')
        i = item(lib)
        album = lib.add_album([i])
        album['flex'] = 'foo'
        album.store()
        self.assertTrue('flex' in i)
        self.assertFalse('flex' in i.keys(with_album=False))
        self.assertEqual(i['flex'], 'foo')
        self.assertEqual(i.get('flex'), 'foo')
        self.assertEqual(i.get('flex', with_album=False), None)
        self.assertEqual(i.get('flexx'), None)

class DestinationTest(_common.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.lib = beets.library.Library(':memory:', '~/.Music')
        self.i = item(self.lib)

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        self.lib._connection().close()
        config.clear()
        config.read(user=False, defaults=True)

    def test_directory_works_with_trailing_slash(self):
        if False:
            for i in range(10):
                print('nop')
        self.lib.directory = b'one/'
        self.lib.path_formats = [('default', 'two')]
        self.assertEqual(self.i.destination(), np('one/two'))

    def test_directory_works_without_trailing_slash(self):
        if False:
            print('Hello World!')
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two')]
        self.assertEqual(self.i.destination(), np('one/two'))

    def test_destination_substitutes_metadata_values(self):
        if False:
            while True:
                i = 10
        self.lib.directory = b'base'
        self.lib.path_formats = [('default', '$album/$artist $title')]
        self.i.title = 'three'
        self.i.artist = 'two'
        self.i.album = 'one'
        self.assertEqual(self.i.destination(), np('base/one/two three'))

    def test_destination_preserves_extension(self):
        if False:
            while True:
                i = 10
        self.lib.directory = b'base'
        self.lib.path_formats = [('default', '$title')]
        self.i.path = 'hey.audioformat'
        self.assertEqual(self.i.destination(), np('base/the title.audioformat'))

    def test_lower_case_extension(self):
        if False:
            while True:
                i = 10
        self.lib.directory = b'base'
        self.lib.path_formats = [('default', '$title')]
        self.i.path = 'hey.MP3'
        self.assertEqual(self.i.destination(), np('base/the title.mp3'))

    def test_destination_pads_some_indices(self):
        if False:
            print('Hello World!')
        self.lib.directory = b'base'
        self.lib.path_formats = [('default', '$track $tracktotal $disc $disctotal $bpm')]
        self.i.track = 1
        self.i.tracktotal = 2
        self.i.disc = 3
        self.i.disctotal = 4
        self.i.bpm = 5
        self.assertEqual(self.i.destination(), np('base/01 02 03 04 5'))

    def test_destination_pads_date_values(self):
        if False:
            return 10
        self.lib.directory = b'base'
        self.lib.path_formats = [('default', '$year-$month-$day')]
        self.i.year = 1
        self.i.month = 2
        self.i.day = 3
        self.assertEqual(self.i.destination(), np('base/0001-02-03'))

    def test_destination_escapes_slashes(self):
        if False:
            while True:
                i = 10
        self.i.album = 'one/two'
        dest = self.i.destination()
        self.assertTrue(b'one' in dest)
        self.assertTrue(b'two' in dest)
        self.assertFalse(b'one/two' in dest)

    def test_destination_escapes_leading_dot(self):
        if False:
            i = 10
            return i + 15
        self.i.album = '.something'
        dest = self.i.destination()
        self.assertTrue(b'something' in dest)
        self.assertFalse(b'/.something' in dest)

    def test_destination_preserves_legitimate_slashes(self):
        if False:
            i = 10
            return i + 15
        self.i.artist = 'one'
        self.i.album = 'two'
        dest = self.i.destination()
        self.assertTrue(os.path.join(b'one', b'two') in dest)

    def test_destination_long_names_truncated(self):
        if False:
            i = 10
            return i + 15
        self.i.title = 'X' * 300
        self.i.artist = 'Y' * 300
        for c in self.i.destination().split(util.PATH_SEP):
            self.assertTrue(len(c) <= 255)

    def test_destination_long_names_keep_extension(self):
        if False:
            return 10
        self.i.title = 'X' * 300
        self.i.path = b'something.extn'
        dest = self.i.destination()
        self.assertEqual(dest[-5:], b'.extn')

    def test_distination_windows_removes_both_separators(self):
        if False:
            while True:
                i = 10
        self.i.title = 'one \\ two / three.mp3'
        with _common.platform_windows():
            p = self.i.destination()
        self.assertFalse(b'one \\ two' in p)
        self.assertFalse(b'one / two' in p)
        self.assertFalse(b'two \\ three' in p)
        self.assertFalse(b'two / three' in p)

    def test_path_with_format(self):
        if False:
            return 10
        self.lib.path_formats = [('default', '$artist/$album ($format)')]
        p = self.i.destination()
        self.assertTrue(b'(FLAC)' in p)

    def test_heterogeneous_album_gets_single_directory(self):
        if False:
            return 10
        (i1, i2) = (item(), item())
        self.lib.add_album([i1, i2])
        (i1.year, i2.year) = (2009, 2010)
        self.lib.path_formats = [('default', '$album ($year)/$track $title')]
        (dest1, dest2) = (i1.destination(), i2.destination())
        self.assertEqual(os.path.dirname(dest1), os.path.dirname(dest2))

    def test_default_path_for_non_compilations(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.comp = False
        self.lib.add_album([self.i])
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('comp:true', 'three')]
        self.assertEqual(self.i.destination(), np('one/two'))

    def test_singleton_path(self):
        if False:
            print('Hello World!')
        i = item(self.lib)
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('singleton:true', 'four'), ('comp:true', 'three')]
        self.assertEqual(i.destination(), np('one/four'))

    def test_comp_before_singleton_path(self):
        if False:
            print('Hello World!')
        i = item(self.lib)
        i.comp = True
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('comp:true', 'three'), ('singleton:true', 'four')]
        self.assertEqual(i.destination(), np('one/three'))

    def test_comp_path(self):
        if False:
            while True:
                i = 10
        self.i.comp = True
        self.lib.add_album([self.i])
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('comp:true', 'three')]
        self.assertEqual(self.i.destination(), np('one/three'))

    def test_albumtype_query_path(self):
        if False:
            i = 10
            return i + 15
        self.i.comp = True
        self.lib.add_album([self.i])
        self.i.albumtype = 'sometype'
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('albumtype:sometype', 'four'), ('comp:true', 'three')]
        self.assertEqual(self.i.destination(), np('one/four'))

    def test_albumtype_path_fallback_to_comp(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.comp = True
        self.lib.add_album([self.i])
        self.i.albumtype = 'sometype'
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('albumtype:anothertype', 'four'), ('comp:true', 'three')]
        self.assertEqual(self.i.destination(), np('one/three'))

    def test_get_formatted_does_not_replace_separators(self):
        if False:
            i = 10
            return i + 15
        with _common.platform_posix():
            name = os.path.join('a', 'b')
            self.i.title = name
            newname = self.i.formatted().get('title')
        self.assertEqual(name, newname)

    def test_get_formatted_pads_with_zero(self):
        if False:
            i = 10
            return i + 15
        with _common.platform_posix():
            self.i.track = 1
            name = self.i.formatted().get('track')
        self.assertTrue(name.startswith('0'))

    def test_get_formatted_uses_kbps_bitrate(self):
        if False:
            for i in range(10):
                print('nop')
        with _common.platform_posix():
            self.i.bitrate = 12345
            val = self.i.formatted().get('bitrate')
        self.assertEqual(val, '12kbps')

    def test_get_formatted_uses_khz_samplerate(self):
        if False:
            i = 10
            return i + 15
        with _common.platform_posix():
            self.i.samplerate = 12345
            val = self.i.formatted().get('samplerate')
        self.assertEqual(val, '12kHz')

    def test_get_formatted_datetime(self):
        if False:
            for i in range(10):
                print('nop')
        with _common.platform_posix():
            self.i.added = 1368302461.210265
            val = self.i.formatted().get('added')
        self.assertTrue(val.startswith('2013'))

    def test_get_formatted_none(self):
        if False:
            print('Hello World!')
        with _common.platform_posix():
            self.i.some_other_field = None
            val = self.i.formatted().get('some_other_field')
        self.assertEqual(val, '')

    def test_artist_falls_back_to_albumartist(self):
        if False:
            i = 10
            return i + 15
        self.i.artist = ''
        self.i.albumartist = 'something'
        self.lib.path_formats = [('default', '$artist')]
        p = self.i.destination()
        self.assertEqual(p.rsplit(util.PATH_SEP, 1)[1], b'something')

    def test_albumartist_falls_back_to_artist(self):
        if False:
            i = 10
            return i + 15
        self.i.artist = 'trackartist'
        self.i.albumartist = ''
        self.lib.path_formats = [('default', '$albumartist')]
        p = self.i.destination()
        self.assertEqual(p.rsplit(util.PATH_SEP, 1)[1], b'trackartist')

    def test_artist_overrides_albumartist(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.artist = 'theartist'
        self.i.albumartist = 'something'
        self.lib.path_formats = [('default', '$artist')]
        p = self.i.destination()
        self.assertEqual(p.rsplit(util.PATH_SEP, 1)[1], b'theartist')

    def test_albumartist_overrides_artist(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.artist = 'theartist'
        self.i.albumartist = 'something'
        self.lib.path_formats = [('default', '$albumartist')]
        p = self.i.destination()
        self.assertEqual(p.rsplit(util.PATH_SEP, 1)[1], b'something')

    def test_unicode_normalized_nfd_on_mac(self):
        if False:
            print('Hello World!')
        instr = unicodedata.normalize('NFC', 'café')
        self.lib.path_formats = [('default', instr)]
        dest = self.i.destination(platform='darwin', fragment=True)
        self.assertEqual(dest, unicodedata.normalize('NFD', instr))

    def test_unicode_normalized_nfc_on_linux(self):
        if False:
            print('Hello World!')
        instr = unicodedata.normalize('NFD', 'café')
        self.lib.path_formats = [('default', instr)]
        dest = self.i.destination(platform='linux', fragment=True)
        self.assertEqual(dest, unicodedata.normalize('NFC', instr))

    def test_non_mbcs_characters_on_windows(self):
        if False:
            for i in range(10):
                print('nop')
        oldfunc = sys.getfilesystemencoding
        sys.getfilesystemencoding = lambda : 'mbcs'
        try:
            self.i.title = 'həd'
            self.lib.path_formats = [('default', '$title')]
            p = self.i.destination()
            self.assertFalse(b'?' in p)
            self.assertTrue('həd'.encode() in p)
        finally:
            sys.getfilesystemencoding = oldfunc

    def test_unicode_extension_in_fragment(self):
        if False:
            print('Hello World!')
        self.lib.path_formats = [('default', 'foo')]
        self.i.path = util.bytestring_path('bar.café')
        dest = self.i.destination(platform='linux', fragment=True)
        self.assertEqual(dest, 'foo.café')

    def test_asciify_and_replace(self):
        if False:
            return 10
        config['asciify_paths'] = True
        self.lib.replacements = [(re.compile('"'), 'q')]
        self.lib.directory = b'lib'
        self.lib.path_formats = [('default', '$title')]
        self.i.title = '“ö—Ï”'
        self.assertEqual(self.i.destination(), np('lib/qo--Iq'))

    def test_asciify_character_expanding_to_slash(self):
        if False:
            print('Hello World!')
        config['asciify_paths'] = True
        self.lib.directory = b'lib'
        self.lib.path_formats = [('default', '$title')]
        self.i.title = 'ab¢½d'
        self.assertEqual(self.i.destination(), np('lib/abC_ 1_2d'))

    def test_destination_with_replacements(self):
        if False:
            return 10
        self.lib.directory = b'base'
        self.lib.replacements = [(re.compile('a'), 'e')]
        self.lib.path_formats = [('default', '$album/$title')]
        self.i.title = 'foo'
        self.i.album = 'bar'
        self.assertEqual(self.i.destination(), np('base/ber/foo'))

    def test_destination_with_replacements_argument(self):
        if False:
            print('Hello World!')
        self.lib.directory = b'base'
        self.lib.replacements = [(re.compile('a'), 'f')]
        self.lib.path_formats = [('default', '$album/$title')]
        self.i.title = 'foo'
        self.i.album = 'bar'
        replacements = [(re.compile('a'), 'e')]
        self.assertEqual(self.i.destination(replacements=replacements), np('base/ber/foo'))

    @unittest.skip('unimplemented: #359')
    def test_destination_with_empty_component(self):
        if False:
            print('Hello World!')
        self.lib.directory = b'base'
        self.lib.replacements = [(re.compile('^$'), '_')]
        self.lib.path_formats = [('default', '$album/$artist/$title')]
        self.i.title = 'three'
        self.i.artist = ''
        self.i.albumartist = ''
        self.i.album = 'one'
        self.assertEqual(self.i.destination(), np('base/one/_/three'))

    @unittest.skip('unimplemented: #359')
    def test_destination_with_empty_final_component(self):
        if False:
            i = 10
            return i + 15
        self.lib.directory = b'base'
        self.lib.replacements = [(re.compile('^$'), '_')]
        self.lib.path_formats = [('default', '$album/$title')]
        self.i.title = ''
        self.i.album = 'one'
        self.i.path = 'foo.mp3'
        self.assertEqual(self.i.destination(), np('base/one/_.mp3'))

    def test_legalize_path_one_for_one_replacement(self):
        if False:
            print('Hello World!')
        self.lib.replacements = [(re.compile('X$'), 'Z')]
        self.i.title = 'X' * 300 + 'Y'
        dest = self.i.destination()
        self.assertEqual(dest[-2:], b'XZ')

    def test_legalize_path_one_for_many_replacement(self):
        if False:
            return 10
        self.lib.replacements = [(re.compile('X$'), 'ZZZZ')]
        self.i.title = 'X' * 300 + 'Y'
        dest = self.i.destination()
        self.assertEqual(dest[-2:], b'XX')

    def test_album_field_query(self):
        if False:
            i = 10
            return i + 15
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', 'two'), ('flex:foo', 'three')]
        album = self.lib.add_album([self.i])
        self.assertEqual(self.i.destination(), np('one/two'))
        album['flex'] = 'foo'
        album.store()
        self.assertEqual(self.i.destination(), np('one/three'))

    def test_album_field_in_template(self):
        if False:
            i = 10
            return i + 15
        self.lib.directory = b'one'
        self.lib.path_formats = [('default', '$flex/two')]
        album = self.lib.add_album([self.i])
        album['flex'] = 'foo'
        album.store()
        self.assertEqual(self.i.destination(), np('one/foo/two'))

class ItemFormattedMappingTest(_common.LibTestCase):

    def test_formatted_item_value(self):
        if False:
            print('Hello World!')
        formatted = self.i.formatted()
        self.assertEqual(formatted['artist'], 'the artist')

    def test_get_unset_field(self):
        if False:
            for i in range(10):
                print('nop')
        formatted = self.i.formatted()
        with self.assertRaises(KeyError):
            formatted['other_field']

    def test_get_method_with_default(self):
        if False:
            i = 10
            return i + 15
        formatted = self.i.formatted()
        self.assertEqual(formatted.get('other_field'), '')

    def test_get_method_with_specified_default(self):
        if False:
            return 10
        formatted = self.i.formatted()
        self.assertEqual(formatted.get('other_field', 'default'), 'default')

    def test_item_precedence(self):
        if False:
            return 10
        album = self.lib.add_album([self.i])
        album['artist'] = 'foo'
        album.store()
        self.assertNotEqual('foo', self.i.formatted().get('artist'))

    def test_album_flex_field(self):
        if False:
            while True:
                i = 10
        album = self.lib.add_album([self.i])
        album['flex'] = 'foo'
        album.store()
        self.assertEqual('foo', self.i.formatted().get('flex'))

    def test_album_field_overrides_item_field_for_path(self):
        if False:
            while True:
                i = 10
        album = self.lib.add_album([self.i])
        album.album = 'foo'
        album.store()
        self.i.album = 'bar'
        self.i.store()
        formatted = self.i.formatted(for_path=True)
        self.assertEqual(formatted['album'], 'foo')

    def test_artist_falls_back_to_albumartist(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.artist = ''
        formatted = self.i.formatted()
        self.assertEqual(formatted['artist'], 'the album artist')

    def test_albumartist_falls_back_to_artist(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.albumartist = ''
        formatted = self.i.formatted()
        self.assertEqual(formatted['albumartist'], 'the artist')

    def test_both_artist_and_albumartist_empty(self):
        if False:
            print('Hello World!')
        self.i.artist = ''
        self.i.albumartist = ''
        formatted = self.i.formatted()
        self.assertEqual(formatted['albumartist'], '')

class PathFormattingMixin:
    """Utilities for testing path formatting."""

    def _setf(self, fmt):
        if False:
            while True:
                i = 10
        self.lib.path_formats.insert(0, ('default', fmt))

    def _assert_dest(self, dest, i=None):
        if False:
            print('Hello World!')
        if i is None:
            i = self.i
        with _common.platform_posix():
            actual = i.destination()
        self.assertEqual(actual, dest)

class DestinationFunctionTest(_common.TestCase, PathFormattingMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.lib = beets.library.Library(':memory:')
        self.lib.directory = b'/base'
        self.lib.path_formats = [('default', 'path')]
        self.i = item(self.lib)

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        self.lib._connection().close()

    def test_upper_case_literal(self):
        if False:
            return 10
        self._setf('%upper{foo}')
        self._assert_dest(b'/base/FOO')

    def test_upper_case_variable(self):
        if False:
            i = 10
            return i + 15
        self._setf('%upper{$title}')
        self._assert_dest(b'/base/THE TITLE')

    def test_title_case_variable(self):
        if False:
            return 10
        self._setf('%title{$title}')
        self._assert_dest(b'/base/The Title')

    def test_title_case_variable_aphostrophe(self):
        if False:
            while True:
                i = 10
        self._setf("%title{I can't}")
        self._assert_dest(b"/base/I Can't")

    def test_asciify_variable(self):
        if False:
            print('Hello World!')
        self._setf('%asciify{ab¢½d}')
        self._assert_dest(b'/base/abC_ 1_2d')

    def test_left_variable(self):
        if False:
            i = 10
            return i + 15
        self._setf('%left{$title, 3}')
        self._assert_dest(b'/base/the')

    def test_right_variable(self):
        if False:
            while True:
                i = 10
        self._setf('%right{$title,3}')
        self._assert_dest(b'/base/tle')

    def test_if_false(self):
        if False:
            print('Hello World!')
        self._setf('x%if{,foo}')
        self._assert_dest(b'/base/x')

    def test_if_false_value(self):
        if False:
            return 10
        self._setf('x%if{false,foo}')
        self._assert_dest(b'/base/x')

    def test_if_true(self):
        if False:
            for i in range(10):
                print('nop')
        self._setf('%if{bar,foo}')
        self._assert_dest(b'/base/foo')

    def test_if_else_false(self):
        if False:
            i = 10
            return i + 15
        self._setf('%if{,foo,baz}')
        self._assert_dest(b'/base/baz')

    def test_if_else_false_value(self):
        if False:
            return 10
        self._setf('%if{false,foo,baz}')
        self._assert_dest(b'/base/baz')

    def test_if_int_value(self):
        if False:
            i = 10
            return i + 15
        self._setf('%if{0,foo,baz}')
        self._assert_dest(b'/base/baz')

    def test_nonexistent_function(self):
        if False:
            while True:
                i = 10
        self._setf('%foo{bar}')
        self._assert_dest(b'/base/%foo{bar}')

    def test_if_def_field_return_self(self):
        if False:
            while True:
                i = 10
        self.i.bar = 3
        self._setf('%ifdef{bar}')
        self._assert_dest(b'/base/3')

    def test_if_def_field_not_defined(self):
        if False:
            print('Hello World!')
        self._setf(' %ifdef{bar}/$artist')
        self._assert_dest(b'/base/the artist')

    def test_if_def_field_not_defined_2(self):
        if False:
            i = 10
            return i + 15
        self._setf('$artist/%ifdef{bar}')
        self._assert_dest(b'/base/the artist')

    def test_if_def_true(self):
        if False:
            i = 10
            return i + 15
        self._setf('%ifdef{artist,cool}')
        self._assert_dest(b'/base/cool')

    def test_if_def_true_complete(self):
        if False:
            while True:
                i = 10
        self.i.series = 'Now'
        self._setf('%ifdef{series,$series Series,Albums}/$album')
        self._assert_dest(b'/base/Now Series/the album')

    def test_if_def_false_complete(self):
        if False:
            for i in range(10):
                print('nop')
        self._setf('%ifdef{plays,$plays,not_played}')
        self._assert_dest(b'/base/not_played')

    def test_first(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.genres = 'Pop; Rock; Classical Crossover'
        self._setf('%first{$genres}')
        self._assert_dest(b'/base/Pop')

    def test_first_skip(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.genres = 'Pop; Rock; Classical Crossover'
        self._setf('%first{$genres,1,2}')
        self._assert_dest(b'/base/Classical Crossover')

    def test_first_different_sep(self):
        if False:
            i = 10
            return i + 15
        self._setf('%first{Alice / Bob / Eve,2,0, / , & }')
        self._assert_dest(b'/base/Alice & Bob')

class DisambiguationTest(_common.TestCase, PathFormattingMixin):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.lib = beets.library.Library(':memory:')
        self.lib.directory = b'/base'
        self.lib.path_formats = [('default', 'path')]
        self.i1 = item()
        self.i1.year = 2001
        self.lib.add_album([self.i1])
        self.i2 = item()
        self.i2.year = 2002
        self.lib.add_album([self.i2])
        self.lib._connection().commit()
        self._setf('foo%aunique{albumartist album,year}/$title')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        self.lib._connection().close()

    def test_unique_expands_to_disambiguating_year(self):
        if False:
            while True:
                i = 10
        self._assert_dest(b'/base/foo [2001]/the title', self.i1)

    def test_unique_with_default_arguments_uses_albumtype(self):
        if False:
            return 10
        album2 = self.lib.get_album(self.i1)
        album2.albumtype = 'bar'
        album2.store()
        self._setf('foo%aunique{}/$title')
        self._assert_dest(b'/base/foo [bar]/the title', self.i1)

    def test_unique_expands_to_nothing_for_distinct_albums(self):
        if False:
            return 10
        album2 = self.lib.get_album(self.i2)
        album2.album = 'different album'
        album2.store()
        self._assert_dest(b'/base/foo/the title', self.i1)

    def test_use_fallback_numbers_when_identical(self):
        if False:
            i = 10
            return i + 15
        album2 = self.lib.get_album(self.i2)
        album2.year = 2001
        album2.store()
        self._assert_dest(b'/base/foo [1]/the title', self.i1)
        self._assert_dest(b'/base/foo [2]/the title', self.i2)

    def test_unique_falls_back_to_second_distinguishing_field(self):
        if False:
            return 10
        self._setf('foo%aunique{albumartist album,month year}/$title')
        self._assert_dest(b'/base/foo [2001]/the title', self.i1)

    def test_unique_sanitized(self):
        if False:
            i = 10
            return i + 15
        album2 = self.lib.get_album(self.i2)
        album2.year = 2001
        album1 = self.lib.get_album(self.i1)
        album1.albumtype = 'foo/bar'
        album2.store()
        album1.store()
        self._setf('foo%aunique{albumartist album,albumtype}/$title')
        self._assert_dest(b'/base/foo [foo_bar]/the title', self.i1)

    def test_drop_empty_disambig_string(self):
        if False:
            while True:
                i = 10
        album1 = self.lib.get_album(self.i1)
        album1.albumdisambig = None
        album2 = self.lib.get_album(self.i2)
        album2.albumdisambig = 'foo'
        album1.store()
        album2.store()
        self._setf('foo%aunique{albumartist album,albumdisambig}/$title')
        self._assert_dest(b'/base/foo/the title', self.i1)

    def test_change_brackets(self):
        if False:
            print('Hello World!')
        self._setf('foo%aunique{albumartist album,year,()}/$title')
        self._assert_dest(b'/base/foo (2001)/the title', self.i1)

    def test_remove_brackets(self):
        if False:
            i = 10
            return i + 15
        self._setf('foo%aunique{albumartist album,year,}/$title')
        self._assert_dest(b'/base/foo 2001/the title', self.i1)

    def test_key_flexible_attribute(self):
        if False:
            while True:
                i = 10
        album1 = self.lib.get_album(self.i1)
        album1.flex = 'flex1'
        album2 = self.lib.get_album(self.i2)
        album2.flex = 'flex2'
        album1.store()
        album2.store()
        self._setf('foo%aunique{albumartist album flex,year}/$title')
        self._assert_dest(b'/base/foo/the title', self.i1)

class SingletonDisambiguationTest(_common.TestCase, PathFormattingMixin):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.lib = beets.library.Library(':memory:')
        self.lib.directory = b'/base'
        self.lib.path_formats = [('default', 'path')]
        self.i1 = item()
        self.i1.year = 2001
        self.lib.add(self.i1)
        self.i2 = item()
        self.i2.year = 2002
        self.lib.add(self.i2)
        self.lib._connection().commit()
        self._setf('foo/$title%sunique{artist title,year}')

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        self.lib._connection().close()

    def test_sunique_expands_to_disambiguating_year(self):
        if False:
            print('Hello World!')
        self._assert_dest(b'/base/foo/the title [2001]', self.i1)

    def test_sunique_with_default_arguments_uses_trackdisambig(self):
        if False:
            return 10
        self.i1.trackdisambig = 'live version'
        self.i1.year = self.i2.year
        self.i1.store()
        self._setf('foo/$title%sunique{}')
        self._assert_dest(b'/base/foo/the title [live version]', self.i1)

    def test_sunique_expands_to_nothing_for_distinct_singletons(self):
        if False:
            while True:
                i = 10
        self.i2.title = 'different track'
        self.i2.store()
        self._assert_dest(b'/base/foo/the title', self.i1)

    def test_sunique_does_not_match_album(self):
        if False:
            for i in range(10):
                print('nop')
        self.lib.add_album([self.i2])
        self._assert_dest(b'/base/foo/the title', self.i1)

    def test_sunique_use_fallback_numbers_when_identical(self):
        if False:
            return 10
        self.i2.year = self.i1.year
        self.i2.store()
        self._assert_dest(b'/base/foo/the title [1]', self.i1)
        self._assert_dest(b'/base/foo/the title [2]', self.i2)

    def test_sunique_falls_back_to_second_distinguishing_field(self):
        if False:
            i = 10
            return i + 15
        self._setf('foo/$title%sunique{albumartist album,month year}')
        self._assert_dest(b'/base/foo/the title [2001]', self.i1)

    def test_sunique_sanitized(self):
        if False:
            while True:
                i = 10
        self.i2.year = self.i1.year
        self.i1.trackdisambig = 'foo/bar'
        self.i2.store()
        self.i1.store()
        self._setf('foo/$title%sunique{artist title,trackdisambig}')
        self._assert_dest(b'/base/foo/the title [foo_bar]', self.i1)

    def test_drop_empty_disambig_string(self):
        if False:
            print('Hello World!')
        self.i1.trackdisambig = None
        self.i2.trackdisambig = 'foo'
        self.i1.store()
        self.i2.store()
        self._setf('foo/$title%sunique{albumartist album,trackdisambig}')
        self._assert_dest(b'/base/foo/the title', self.i1)

    def test_change_brackets(self):
        if False:
            while True:
                i = 10
        self._setf('foo/$title%sunique{artist title,year,()}')
        self._assert_dest(b'/base/foo/the title (2001)', self.i1)

    def test_remove_brackets(self):
        if False:
            while True:
                i = 10
        self._setf('foo/$title%sunique{artist title,year,}')
        self._assert_dest(b'/base/foo/the title 2001', self.i1)

    def test_key_flexible_attribute(self):
        if False:
            while True:
                i = 10
        self.i1.flex = 'flex1'
        self.i2.flex = 'flex2'
        self.i1.store()
        self.i2.store()
        self._setf('foo/$title%sunique{artist title flex,year}')
        self._assert_dest(b'/base/foo/the title', self.i1)

class PluginDestinationTest(_common.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._tv_map = {}

        def field_getters():
            if False:
                print('Hello World!')
            getters = {}
            for (key, value) in self._tv_map.items():
                getters[key] = lambda _: value
            return getters
        self.old_field_getters = plugins.item_field_getters
        plugins.item_field_getters = field_getters
        self.lib = beets.library.Library(':memory:')
        self.lib.directory = b'/base'
        self.lib.path_formats = [('default', '$artist $foo')]
        self.i = item(self.lib)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        plugins.item_field_getters = self.old_field_getters

    def _assert_dest(self, dest):
        if False:
            i = 10
            return i + 15
        with _common.platform_posix():
            the_dest = self.i.destination()
        self.assertEqual(the_dest, b'/base/' + dest)

    def test_undefined_value_not_substituted(self):
        if False:
            i = 10
            return i + 15
        self._assert_dest(b'the artist $foo')

    def test_plugin_value_not_substituted(self):
        if False:
            while True:
                i = 10
        self._tv_map = {'foo': 'bar'}
        self._assert_dest(b'the artist bar')

    def test_plugin_value_overrides_attribute(self):
        if False:
            print('Hello World!')
        self._tv_map = {'artist': 'bar'}
        self._assert_dest(b'bar $foo')

    def test_plugin_value_sanitized(self):
        if False:
            print('Hello World!')
        self._tv_map = {'foo': 'bar/baz'}
        self._assert_dest(b'the artist bar_baz')

class AlbumInfoTest(_common.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.lib = beets.library.Library(':memory:')
        self.i = item()
        self.lib.add_album((self.i,))

    def test_albuminfo_reflects_metadata(self):
        if False:
            i = 10
            return i + 15
        ai = self.lib.get_album(self.i)
        self.assertEqual(ai.mb_albumartistid, self.i.mb_albumartistid)
        self.assertEqual(ai.albumartist, self.i.albumartist)
        self.assertEqual(ai.album, self.i.album)
        self.assertEqual(ai.year, self.i.year)

    def test_albuminfo_stores_art(self):
        if False:
            print('Hello World!')
        ai = self.lib.get_album(self.i)
        ai.artpath = '/my/great/art'
        ai.store()
        new_ai = self.lib.get_album(self.i)
        self.assertEqual(new_ai.artpath, b'/my/great/art')

    def test_albuminfo_for_two_items_doesnt_duplicate_row(self):
        if False:
            i = 10
            return i + 15
        i2 = item(self.lib)
        self.lib.get_album(self.i)
        self.lib.get_album(i2)
        c = self.lib._connection().cursor()
        c.execute('select * from albums where album=?', (self.i.album,))
        self.assertNotEqual(c.fetchone(), None)
        self.assertEqual(c.fetchone(), None)

    def test_individual_tracks_have_no_albuminfo(self):
        if False:
            return 10
        i2 = item()
        i2.album = 'aTotallyDifferentAlbum'
        self.lib.add(i2)
        ai = self.lib.get_album(i2)
        self.assertEqual(ai, None)

    def test_get_album_by_id(self):
        if False:
            for i in range(10):
                print('nop')
        ai = self.lib.get_album(self.i)
        ai = self.lib.get_album(self.i.id)
        self.assertNotEqual(ai, None)

    def test_album_items_consistent(self):
        if False:
            return 10
        ai = self.lib.get_album(self.i)
        for i in ai.items():
            if i.id == self.i.id:
                break
        else:
            self.fail('item not found')

    def test_albuminfo_changes_affect_items(self):
        if False:
            return 10
        ai = self.lib.get_album(self.i)
        ai.album = 'myNewAlbum'
        ai.store()
        i = self.lib.items()[0]
        self.assertEqual(i.album, 'myNewAlbum')

    def test_albuminfo_change_albumartist_changes_items(self):
        if False:
            while True:
                i = 10
        ai = self.lib.get_album(self.i)
        ai.albumartist = 'myNewArtist'
        ai.store()
        i = self.lib.items()[0]
        self.assertEqual(i.albumartist, 'myNewArtist')
        self.assertNotEqual(i.artist, 'myNewArtist')

    def test_albuminfo_change_artist_does_change_items(self):
        if False:
            i = 10
            return i + 15
        ai = self.lib.get_album(self.i)
        ai.artist = 'myNewArtist'
        ai.store(inherit=True)
        i = self.lib.items()[0]
        self.assertEqual(i.artist, 'myNewArtist')

    def test_albuminfo_change_artist_does_not_change_items(self):
        if False:
            while True:
                i = 10
        ai = self.lib.get_album(self.i)
        ai.artist = 'myNewArtist'
        ai.store(inherit=False)
        i = self.lib.items()[0]
        self.assertNotEqual(i.artist, 'myNewArtist')

    def test_albuminfo_remove_removes_items(self):
        if False:
            while True:
                i = 10
        item_id = self.i.id
        self.lib.get_album(self.i).remove()
        c = self.lib._connection().execute('SELECT id FROM items WHERE id=?', (item_id,))
        self.assertEqual(c.fetchone(), None)

    def test_removing_last_item_removes_album(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(self.lib.albums()), 1)
        self.i.remove()
        self.assertEqual(len(self.lib.albums()), 0)

    def test_noop_albuminfo_changes_affect_items(self):
        if False:
            for i in range(10):
                print('nop')
        i = self.lib.items()[0]
        i.album = 'foobar'
        i.store()
        ai = self.lib.get_album(self.i)
        ai.album = ai.album
        ai.store()
        i = self.lib.items()[0]
        self.assertEqual(i.album, ai.album)

class ArtDestinationTest(_common.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        config['art_filename'] = 'artimage'
        config['replace'] = {'X': 'Y'}
        self.lib = beets.library.Library(':memory:', replacements=[(re.compile('X'), 'Y')])
        self.i = item(self.lib)
        self.i.path = self.i.destination()
        self.ai = self.lib.add_album((self.i,))

    def test_art_filename_respects_setting(self):
        if False:
            print('Hello World!')
        art = self.ai.art_destination('something.jpg')
        new_art = bytestring_path('%sartimage.jpg' % os.path.sep)
        self.assertTrue(new_art in art)

    def test_art_path_in_item_dir(self):
        if False:
            for i in range(10):
                print('nop')
        art = self.ai.art_destination('something.jpg')
        track = self.i.destination()
        self.assertEqual(os.path.dirname(art), os.path.dirname(track))

    def test_art_path_sanitized(self):
        if False:
            while True:
                i = 10
        config['art_filename'] = 'artXimage'
        art = self.ai.art_destination('something.jpg')
        self.assertTrue(b'artYimage' in art)

class PathStringTest(_common.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.lib = beets.library.Library(':memory:')
        self.i = item(self.lib)

    def test_item_path_is_bytestring(self):
        if False:
            return 10
        self.assertTrue(isinstance(self.i.path, bytes))

    def test_fetched_item_path_is_bytestring(self):
        if False:
            while True:
                i = 10
        i = list(self.lib.items())[0]
        self.assertTrue(isinstance(i.path, bytes))

    def test_unicode_path_becomes_bytestring(self):
        if False:
            return 10
        self.i.path = 'unicodepath'
        self.assertTrue(isinstance(self.i.path, bytes))

    def test_unicode_in_database_becomes_bytestring(self):
        if False:
            return 10
        self.lib._connection().execute('\n        update items set path=? where id=?\n        ', (self.i.id, 'somepath'))
        i = list(self.lib.items())[0]
        self.assertTrue(isinstance(i.path, bytes))

    def test_special_chars_preserved_in_database(self):
        if False:
            while True:
                i = 10
        path = 'bár'.encode()
        self.i.path = path
        self.i.store()
        i = list(self.lib.items())[0]
        self.assertEqual(i.path, path)

    def test_special_char_path_added_to_database(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.remove()
        path = 'bár'.encode()
        i = item()
        i.path = path
        self.lib.add(i)
        i = list(self.lib.items())[0]
        self.assertEqual(i.path, path)

    def test_destination_returns_bytestring(self):
        if False:
            print('Hello World!')
        self.i.artist = 'bár'
        dest = self.i.destination()
        self.assertTrue(isinstance(dest, bytes))

    def test_art_destination_returns_bytestring(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.artist = 'bár'
        alb = self.lib.add_album([self.i])
        dest = alb.art_destination('image.jpg')
        self.assertTrue(isinstance(dest, bytes))

    def test_artpath_stores_special_chars(self):
        if False:
            while True:
                i = 10
        path = b'b\xe1r'
        alb = self.lib.add_album([self.i])
        alb.artpath = path
        alb.store()
        alb = self.lib.get_album(self.i)
        self.assertEqual(path, alb.artpath)

    def test_sanitize_path_with_special_chars(self):
        if False:
            while True:
                i = 10
        path = 'bár?'
        new_path = util.sanitize_path(path)
        self.assertTrue(new_path.startswith('bár'))

    def test_sanitize_path_returns_unicode(self):
        if False:
            return 10
        path = 'bár?'
        new_path = util.sanitize_path(path)
        self.assertTrue(isinstance(new_path, str))

    def test_unicode_artpath_becomes_bytestring(self):
        if False:
            while True:
                i = 10
        alb = self.lib.add_album([self.i])
        alb.artpath = 'somepáth'
        self.assertTrue(isinstance(alb.artpath, bytes))

    def test_unicode_artpath_in_database_decoded(self):
        if False:
            while True:
                i = 10
        alb = self.lib.add_album([self.i])
        self.lib._connection().execute('update albums set artpath=? where id=?', ('somepáth', alb.id))
        alb = self.lib.get_album(alb.id)
        self.assertTrue(isinstance(alb.artpath, bytes))

class MtimeTest(_common.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.ipath = os.path.join(self.temp_dir, b'testfile.mp3')
        shutil.copy(syspath(os.path.join(_common.RSRC, b'full.mp3')), syspath(self.ipath))
        self.i = beets.library.Item.from_path(self.ipath)
        self.lib = beets.library.Library(':memory:')
        self.lib.add(self.i)

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        if os.path.exists(self.ipath):
            os.remove(self.ipath)

    def _mtime(self):
        if False:
            return 10
        return int(os.path.getmtime(self.ipath))

    def test_mtime_initially_up_to_date(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertGreaterEqual(self.i.mtime, self._mtime())

    def test_mtime_reset_on_db_modify(self):
        if False:
            i = 10
            return i + 15
        self.i.title = 'something else'
        self.assertLess(self.i.mtime, self._mtime())

    def test_mtime_up_to_date_after_write(self):
        if False:
            return 10
        self.i.title = 'something else'
        self.i.write()
        self.assertGreaterEqual(self.i.mtime, self._mtime())

    def test_mtime_up_to_date_after_read(self):
        if False:
            return 10
        self.i.title = 'something else'
        self.i.read()
        self.assertGreaterEqual(self.i.mtime, self._mtime())

class ImportTimeTest(_common.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.lib = beets.library.Library(':memory:')

    def added(self):
        if False:
            return 10
        self.track = item()
        self.album = self.lib.add_album((self.track,))
        self.assertGreater(self.album.added, 0)
        self.assertGreater(self.track.added, 0)

    def test_atime_for_singleton(self):
        if False:
            print('Hello World!')
        self.singleton = item(self.lib)
        self.assertGreater(self.singleton.added, 0)

class TemplateTest(_common.LibTestCase):

    def test_year_formatted_in_template(self):
        if False:
            i = 10
            return i + 15
        self.i.year = 123
        self.i.store()
        self.assertEqual(self.i.evaluate_template('$year'), '0123')

    def test_album_flexattr_appears_in_item_template(self):
        if False:
            for i in range(10):
                print('nop')
        self.album = self.lib.add_album([self.i])
        self.album.foo = 'baz'
        self.album.store()
        self.assertEqual(self.i.evaluate_template('$foo'), 'baz')

    def test_album_and_item_format(self):
        if False:
            for i in range(10):
                print('nop')
        config['format_album'] = 'foö $foo'
        album = beets.library.Album()
        album.foo = 'bar'
        album.tagada = 'togodo'
        self.assertEqual(f'{album}', 'foö bar')
        self.assertEqual(f'{album:$tagada}', 'togodo')
        self.assertEqual(str(album), 'foö bar')
        self.assertEqual(bytes(album), b'fo\xc3\xb6 bar')
        config['format_item'] = 'bar $foo'
        item = beets.library.Item()
        item.foo = 'bar'
        item.tagada = 'togodo'
        self.assertEqual(f'{item}', 'bar bar')
        self.assertEqual(f'{item:$tagada}', 'togodo')

class UnicodePathTest(_common.LibTestCase):

    def test_unicode_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.i.path = os.path.join(_common.RSRC, 'unicode’d.mp3'.encode())
        self.i.read()
        self.i.write()

class WriteTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_beets()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.teardown_beets()

    def test_write_nonexistant(self):
        if False:
            i = 10
            return i + 15
        item = self.create_item()
        item.path = b'/path/does/not/exist'
        with self.assertRaises(beets.library.ReadError):
            item.write()

    def test_no_write_permission(self):
        if False:
            for i in range(10):
                print('nop')
        item = self.add_item_fixture()
        path = syspath(item.path)
        os.chmod(path, stat.S_IRUSR)
        try:
            self.assertRaises(beets.library.WriteError, item.write)
        finally:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def test_write_with_custom_path(self):
        if False:
            print('Hello World!')
        item = self.add_item_fixture()
        custom_path = os.path.join(self.temp_dir, b'custom.mp3')
        shutil.copy(syspath(item.path), syspath(custom_path))
        item['artist'] = 'new artist'
        self.assertNotEqual(MediaFile(syspath(custom_path)).artist, 'new artist')
        self.assertNotEqual(MediaFile(syspath(item.path)).artist, 'new artist')
        item.write(custom_path)
        self.assertEqual(MediaFile(syspath(custom_path)).artist, 'new artist')
        self.assertNotEqual(MediaFile(syspath(item.path)).artist, 'new artist')

    def test_write_custom_tags(self):
        if False:
            for i in range(10):
                print('nop')
        item = self.add_item_fixture(artist='old artist')
        item.write(tags={'artist': 'new artist'})
        self.assertNotEqual(item.artist, 'new artist')
        self.assertEqual(MediaFile(syspath(item.path)).artist, 'new artist')

    def test_write_multi_tags(self):
        if False:
            i = 10
            return i + 15
        item = self.add_item_fixture(artist='old artist')
        item.write(tags={'artists': ['old artist', 'another artist']})
        self.assertEqual(MediaFile(syspath(item.path)).artists, ['old artist', 'another artist'])

    def test_write_multi_tags_id3v23(self):
        if False:
            return 10
        item = self.add_item_fixture(artist='old artist')
        item.write(tags={'artists': ['old artist', 'another artist']}, id3v23=True)
        self.assertEqual(MediaFile(syspath(item.path)).artists, ['old artist/another artist'])

    def test_write_date_field(self):
        if False:
            while True:
                i = 10
        item = self.add_item_fixture()
        clean_year = item.year
        item.date = 'foo'
        item.write()
        self.assertEqual(MediaFile(syspath(item.path)).year, clean_year)

class ItemReadTest(unittest.TestCase):

    def test_unreadable_raise_read_error(self):
        if False:
            while True:
                i = 10
        unreadable = os.path.join(_common.RSRC, b'image-2x3.png')
        item = beets.library.Item()
        with self.assertRaises(beets.library.ReadError) as cm:
            item.read(unreadable)
        self.assertIsInstance(cm.exception.reason, UnreadableFileError)

    def test_nonexistent_raise_read_error(self):
        if False:
            while True:
                i = 10
        item = beets.library.Item()
        with self.assertRaises(beets.library.ReadError):
            item.read('/thisfiledoesnotexist')

class FilesizeTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_beets()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.teardown_beets()

    def test_filesize(self):
        if False:
            return 10
        item = self.add_item_fixture()
        self.assertNotEqual(item.filesize, 0)

    def test_nonexistent_file(self):
        if False:
            i = 10
            return i + 15
        item = beets.library.Item()
        self.assertEqual(item.filesize, 0)

class ParseQueryTest(unittest.TestCase):

    def test_parse_invalid_query_string(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(beets.dbcore.InvalidQueryError) as raised:
            beets.library.parse_query_string('foo"', None)
        self.assertIsInstance(raised.exception, beets.dbcore.query.ParsingError)

    def test_parse_bytes(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(AssertionError):
            beets.library.parse_query_string(b'query', None)

class LibraryFieldTypesTest(unittest.TestCase):
    """Test format() and parse() for library-specific field types"""

    def test_datetype(self):
        if False:
            print('Hello World!')
        t = beets.library.DateType()
        time_format = beets.config['time_format'].as_str()
        time_local = time.strftime(time_format, time.localtime(123456789))
        self.assertEqual(time_local, t.format(123456789))
        self.assertEqual(123456789.0, t.parse(time_local))
        self.assertEqual(123456789.0, t.parse('123456789.0'))
        self.assertEqual(t.null, t.parse('not123456789.0'))
        self.assertEqual(t.null, t.parse('1973-11-29'))

    def test_pathtype(self):
        if False:
            print('Hello World!')
        t = beets.library.PathType()
        self.assertEqual('/tmp', t.format('/tmp'))
        self.assertEqual('/tmp/älbum', t.format('/tmp/älbum'))
        self.assertEqual(np(b'/tmp'), t.parse('/tmp'))
        self.assertEqual(np(b'/tmp/\xc3\xa4lbum'), t.parse('/tmp/älbum/'))

    def test_musicalkey(self):
        if False:
            print('Hello World!')
        t = beets.library.MusicalKey()
        self.assertEqual('C#m', t.parse('c#m'))
        self.assertEqual('Gm', t.parse('g   minor'))
        self.assertEqual('Not c#m', t.parse('not C#m'))

    def test_durationtype(self):
        if False:
            return 10
        t = beets.library.DurationType()
        self.assertEqual('1:01', t.format(61.23))
        self.assertEqual('60:01', t.format(3601.23))
        self.assertEqual('0:00', t.format(None))
        self.assertEqual(61.0, t.parse('1:01'))
        self.assertEqual(61.23, t.parse('61.23'))
        self.assertEqual(3601.0, t.parse('60:01'))
        self.assertEqual(t.null, t.parse('1:00:01'))
        self.assertEqual(t.null, t.parse('not61.23'))
        beets.config['format_raw_length'] = True
        self.assertEqual(61.23, t.format(61.23))
        self.assertEqual(3601.23, t.format(3601.23))

def suite():
    if False:
        i = 10
        return i + 15
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')