import unittest
from test.picardtestcase import PicardTestCase
from picard import config
from picard.const.sys import IS_WIN
from picard.file import File
from picard.metadata import Metadata
from picard.script import register_script_function
from picard.util.scripttofilename import script_to_filename, script_to_filename_with_metadata
settings = {'ascii_filenames': False, 'enabled_plugins': [], 'windows_compatibility': False, 'win_compat_replacements': {}, 'replace_spaces_with_underscores': False, 'replace_dir_separator': '_'}

def func_has_file(parser):
    if False:
        return 10
    return '1' if parser.file else ''
register_script_function(lambda p: '1' if p.file else '', 'has_file')

class ScriptToFilenameTest(PicardTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.set_config_values(settings)

    def test_plain_filename(self):
        if False:
            while True:
                i = 10
        metadata = Metadata()
        filename = script_to_filename('AlbumArt', metadata)
        self.assertEqual('AlbumArt', filename)

    def test_simple_script(self):
        if False:
            i = 10
            return i + 15
        metadata = Metadata()
        metadata['artist'] = 'AC/DC'
        metadata['album'] = 'The Album'
        filename = script_to_filename('%album%', metadata)
        self.assertEqual('The Album', filename)
        filename = script_to_filename('%artist%/%album%', metadata)
        self.assertEqual('AC_DC/The Album', filename)

    def test_preserve_backslash(self):
        if False:
            i = 10
            return i + 15
        metadata = Metadata()
        metadata['artist'] = 'AC\\/DC'
        filename = script_to_filename('%artist%', metadata)
        self.assertEqual('AC__DC' if IS_WIN else 'AC\\_DC', filename)

    def test_file_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        metadata = Metadata()
        file = File('somepath/somefile.mp3')
        self.assertEqual('', script_to_filename('$has_file()', metadata))
        self.assertEqual('1', script_to_filename('$has_file()', metadata, file=file))

    def test_script_to_filename_with_metadata(self):
        if False:
            while True:
                i = 10
        metadata = Metadata()
        metadata['artist'] = 'Foo'
        metadata['~extension'] = 'foo'
        (filename, new_metadata) = script_to_filename_with_metadata('$set(_extension,bar)\n%artist%', metadata)
        self.assertEqual('Foo', filename)
        self.assertEqual('foo', metadata['~extension'])
        self.assertEqual('bar', new_metadata['~extension'])

    def test_ascii_filenames(self):
        if False:
            for i in range(10):
                print('nop')
        metadata = Metadata()
        metadata['artist'] = 'Die Ärzte'
        settings = config.setting.copy()
        settings['ascii_filenames'] = False
        filename = script_to_filename('%artist% éöü½', metadata, settings=settings)
        self.assertEqual('Die Ärzte éöü½', filename)
        settings['ascii_filenames'] = True
        filename = script_to_filename('%artist% éöü½', metadata, settings=settings)
        self.assertEqual('Die Arzte eou 1_2', filename)

    def test_windows_compatibility(self):
        if False:
            i = 10
            return i + 15
        metadata = Metadata()
        metadata['artist'] = '\\*:'
        settings = config.setting.copy()
        settings['windows_compatibility'] = False
        expect_orig = '\\*:?'
        expect_compat = '____'
        filename = script_to_filename('%artist%?', metadata, settings=settings)
        self.assertEqual(expect_compat if IS_WIN else expect_orig, filename)
        settings['windows_compatibility'] = True
        filename = script_to_filename('%artist%?', metadata, settings=settings)
        self.assertEqual(expect_compat, filename)

    def test_windows_compatibility_custom_replacements(self):
        if False:
            for i in range(10):
                print('nop')
        metadata = Metadata()
        metadata['artist'] = '\\*:'
        expect_compat = '_+_!'
        settings = config.setting.copy()
        settings['windows_compatibility'] = True
        settings['win_compat_replacements'] = {'*': '+', '?': '!'}
        filename = script_to_filename('%artist%?', metadata, settings=settings)
        self.assertEqual(expect_compat, filename)

    def test_replace_spaces_with_underscores(self):
        if False:
            return 10
        metadata = Metadata()
        metadata['artist'] = ' The \t  New* _ Artist  '
        settings = config.setting.copy()
        settings['windows_compatibility'] = True
        settings['replace_spaces_with_underscores'] = False
        filename = script_to_filename('%artist%', metadata, settings=settings)
        self.assertEqual(' The \t  New_ _ Artist  ', filename)
        settings['replace_spaces_with_underscores'] = True
        filename = script_to_filename('%artist%', metadata, settings=settings)
        self.assertEqual('The_New_Artist', filename)

    def test_replace_dir_separator(self):
        if False:
            for i in range(10):
                print('nop')
        metadata = Metadata()
        metadata['artist'] = 'AC/DC'
        settings = config.setting.copy()
        settings['replace_dir_separator'] = '-'
        filename = script_to_filename('/music/%artist%', metadata, settings=settings)
        self.assertEqual('/music/AC-DC', filename)

    @unittest.skipUnless(IS_WIN, 'windows test')
    def test_ascii_win_save(self):
        if False:
            print('Hello World!')
        self._test_ascii_windows_compatibility()

    def test_ascii_win_compat(self):
        if False:
            while True:
                i = 10
        config.setting['windows_compatibility'] = True
        self._test_ascii_windows_compatibility()

    def _test_ascii_windows_compatibility(self):
        if False:
            return 10
        metadata = Metadata()
        metadata['artist'] = '∖/\\∕'
        settings = config.setting.copy()
        settings['ascii_filenames'] = True
        filename = script_to_filename('%artist%/∖\\\\∕', metadata, settings=settings)
        self.assertEqual('____/_\\_', filename)

    def test_remove_null_chars(self):
        if False:
            return 10
        metadata = Metadata()
        filename = script_to_filename('a\x00b\x00', metadata)
        self.assertEqual('ab', filename)

    def test_remove_tabs_and_linebreaks_chars(self):
        if False:
            return 10
        metadata = Metadata()
        filename = script_to_filename('a\tb\nc', metadata)
        self.assertEqual('abc', filename)

    def test_remove_leading_and_trailing_whitespace(self):
        if False:
            while True:
                i = 10
        metadata = Metadata()
        metadata['artist'] = 'The Artist'
        filename = script_to_filename(' %artist% ', metadata)
        self.assertEqual(' The Artist ', filename)