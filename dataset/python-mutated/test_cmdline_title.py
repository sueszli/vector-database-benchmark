import pytest
from tests.cli.test_cmdline import CommandLineTestCase

@pytest.mark.posix_only()
class TestCommandLineWithTitlePOSIX(CommandLineTestCase):

    def test_open_player_with_title_vlc(self):
        if False:
            while True:
                i = 10
        self._test_args(['streamlink', '-p', '/usr/bin/vlc', '--title', '{title} - {author} - {category}', 'http://test.se', 'test'], ['/usr/bin/vlc', '--input-title-format', 'Test Title - Tѥst Āuƭhǿr - No Category', '-'])

    def test_open_player_with_default_title_vlc(self):
        if False:
            return 10
        self._test_args(['streamlink', '-p', '/usr/bin/vlc', 'http://test.se', 'test'], ['/usr/bin/vlc', '--input-title-format', 'http://test.se', '-'])

    def test_open_player_with_default_title_vlc_args(self):
        if False:
            i = 10
            return i + 15
        self._test_args(['streamlink', '-p', '/Applications/VLC/vlc', '-a', '--other-option', 'http://test.se', 'test'], ['/Applications/VLC/vlc', '--input-title-format', 'http://test.se', '--other-option', '-'])

    def test_open_player_with_title_mpv(self):
        if False:
            i = 10
            return i + 15
        self._test_args(['streamlink', '-p', '/usr/bin/mpv', '--title', '{title}', 'http://test.se', 'test'], ['/usr/bin/mpv', '--force-media-title=Test Title', '-'])

@pytest.mark.windows_only()
class TestCommandLineWithTitleWindows(CommandLineTestCase):

    def test_open_player_with_title_vlc(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_args(['streamlink', '-p', 'c:\\Program Files\\VideoLAN\\vlc.exe', '--title', '{title} - {author} - {category}', 'http://test.se', 'test'], ['c:\\Program Files\\VideoLAN\\vlc.exe', '--input-title-format', 'Test Title - Tѥst Āuƭhǿr - No Category', '-'])

    def test_open_player_with_default_title_vlc(self):
        if False:
            return 10
        self._test_args(['streamlink', '-p', 'c:\\Program Files\\VideoLAN\\vlc.exe', 'http://test.se', 'test'], ['c:\\Program Files\\VideoLAN\\vlc.exe', '--input-title-format', 'http://test.se', '-'])

    def test_open_player_with_default_arg_vlc(self):
        if False:
            i = 10
            return i + 15
        self._test_args(['streamlink', '-p', 'c:\\Program Files\\VideoLAN\\vlc.exe', '-a', '--other-option', 'http://test.se', 'test'], ['c:\\Program Files\\VideoLAN\\vlc.exe', '--input-title-format', 'http://test.se', '--other-option', '-'])

    def test_open_player_with_title_pot(self):
        if False:
            print('Hello World!')
        self._test_args(['streamlink', '--player-passthrough', 'hls', '-p', 'c:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe', '--title', '{title}', 'http://test.se/stream', 'hls'], ['c:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe', 'http://test.se/playlist.m3u8\\Test Title'], passthrough=True)

    def test_open_player_with_default_title_pot(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_args(['streamlink', '--player-passthrough', 'hls', '-p', 'c:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe', 'http://test.se/stream', 'hls'], ['c:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe', 'http://test.se/playlist.m3u8\\http://test.se/stream'], passthrough=True)