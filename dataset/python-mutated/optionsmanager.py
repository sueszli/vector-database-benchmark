"""Youtubedlg module to handle settings. """
from __future__ import unicode_literals
import os
import json
from .utils import os_path_expanduser, os_path_exists, encode_tuple, decode_tuple, check_path, get_default_lang
from .formats import OUTPUT_FORMATS, FORMATS

class OptionsManager(object):
    """Handles youtubedlg options.

    This class is responsible for storing and retrieving the options.

    Attributes:
        SETTINGS_FILENAME (string): Filename of the settings file.
        SENSITIVE_KEYS (tuple): Contains the keys that we don't want
            to store on the settings file. (SECURITY ISSUES).

    Args:
        config_path (string): Absolute path where OptionsManager
            should store the settings file.

    Note:
        See load_default() method for available options.

    Example:
        Access the options using the 'options' variable.

        opt_manager = OptionsManager('.')
        opt_manager.options['save_path'] = '~/Downloads'

    """
    SETTINGS_FILENAME = 'settings.json'
    SENSITIVE_KEYS = ('sudo_password', 'password', 'video_password')

    def __init__(self, config_path):
        if False:
            print('Hello World!')
        self.config_path = config_path
        self.settings_file = os.path.join(config_path, self.SETTINGS_FILENAME)
        self.options = dict()
        self.load_default()
        self.load_from_file()

    def load_default(self):
        if False:
            while True:
                i = 10
        'Load the default options.\n\n        Note:\n            This method is automatically called by the constructor.\n\n        Options Description:\n\n            save_path (string): Path where youtube-dl should store the\n                downloaded file. Default is $HOME.\n\n            video_format (string): Video format to download.\n                When this options is set to \'0\' youtube-dl will choose\n                the best video format available for the given URL.\n\n            second_video_format (string): Video format to mix with the first\n                one (-f 18+17).\n\n            to_audio (boolean): If True youtube-dl will post process the\n                video file.\n\n            keep_video (boolen): If True youtube-dl will keep the video file\n                after post processing it.\n\n            audio_format (string): Audio format of the post processed file.\n                Available values are "mp3", "wav", "aac", "m4a", "vorbis",\n                "opus" & "flac".\n\n            audio_quality (string): Audio quality of the post processed file.\n                Available values are "9", "5", "0". The lowest the value the\n                better the quality.\n\n            restrict_filenames (boolean): If True youtube-dl will restrict\n                the downloaded file filename to ASCII characters only.\n\n            output_format (int): This option sets the downloaded file\n                output template. See formats.OUTPUT_FORMATS for more info.\n\n            output_template (string): Can be any output template supported\n                by youtube-dl.\n\n            playlist_start (int): Playlist index to start downloading.\n\n            playlist_end (int): Playlist index to stop downloading.\n\n            max_downloads (int): Maximum number of video files to download\n                from the given playlist.\n\n            min_filesize (float): Minimum file size of the video file.\n                If the video file is smaller than the given size then\n                youtube-dl will abort the download process.\n\n            max_filesize (float): Maximum file size of the video file.\n                If the video file is larger than the given size then\n                youtube-dl will abort the download process.\n\n            min_filesize_unit (string): Minimum file size unit.\n                Available values: \'\', \'k\', \'m\', \'g\', \'y\', \'p\', \'e\', \'z\', \'y\'.\n\n            max_filesize_unit (string): Maximum file size unit.\n                See \'min_filesize_unit\' option for available values.\n\n            write_subs (boolean): If True youtube-dl will try to download\n                the subtitles file for the given URL.\n\n            write_all_subs (boolean): If True youtube-dl will try to download\n                all the available subtitles files for the given URL.\n\n            write_auto_subs (boolean): If True youtube-dl will try to download\n                the automatic subtitles file for the given URL.\n\n            embed_subs (boolean): If True youtube-dl will merge the subtitles\n                file with the video. (ONLY mp4 files).\n\n            subs_lang (string): Language of the subtitles file to download.\n                Needs \'write_subs\' option.\n\n            ignore_errors (boolean): If True youtube-dl will ignore the errors\n                and continue the download process.\n\n            open_dl_dir (boolean): If True youtube-dlg will open the\n                destination folder after download process has been completed.\n\n            write_description (boolean): If True youtube-dl will write video\n                description to a .description file.\n\n            write_info (boolean): If True youtube-dl will write video\n                metadata to a .info.json file.\n\n            write_thumbnail (boolean): If True youtube-dl will write\n                thumbnail image to disk.\n\n            retries (int): Number of youtube-dl retries.\n\n            user_agent (string): Specify a custom user agent for youtube-dl.\n\n            referer (string): Specify a custom referer to use if the video\n                access is restricted to one domain.\n\n            proxy (string): Use the specified HTTP/HTTPS proxy.\n\n            shutdown (boolean): If True youtube-dlg will turn the computer\n                off after the download process has been completed.\n\n            sudo_password (string): SUDO password for the shutdown process if\n                the user does not have elevated privileges.\n\n            username (string): Username to login with.\n\n            password (string): Password to login with.\n\n            video_password (string): Video password for the given URL.\n\n            youtubedl_path (string): Absolute path to the youtube-dl binary.\n                Default is the self.config_path. You can change this option\n                to point on /usr/local/bin etc.. if you want to use the\n                youtube-dl binary on your system. This is also the directory\n                where youtube-dlg will auto download the youtube-dl if not\n                exists so you should make sure you have write access if you\n                want to update the youtube-dl binary from within youtube-dlg.\n\n            cmd_args (string): String that contains extra youtube-dl options\n                seperated by spaces.\n\n            enable_log (boolean): If True youtube-dlg will enable\n                the LogManager. See main() function under __init__().\n\n            log_time (boolean): See logmanager.LogManager add_time attribute.\n\n            workers_number (int): Number of download workers that download manager\n                will spawn. Must be greater than zero.\n\n            locale_name (string): Locale name (e.g. ru_RU).\n\n            main_win_size (tuple): Main window size (width, height).\n                If window becomes to small the program will reset its size.\n                See _settings_are_valid method MIN_FRAME_SIZE.\n\n            opts_win_size (tuple): Options window size (width, height).\n                If window becomes to small the program will reset its size.\n                See _settings_are_valid method MIN_FRAME_SIZE.\n\n            save_path_dirs (list): List that contains temporary save paths.\n\n            selected_video_formats (list): List that contains the selected\n                video formats to display on the main window.\n\n            selected_audio_formats (list): List that contains the selected\n                audio formats to display on the main window.\n\n            selected_format (string): Current format selected on the main window.\n\n            youtube_dl_debug (boolean): When True will pass \'-v\' flag to youtube-dl.\n\n            ignore_config (boolean): When True will ignore youtube-dl config file options.\n\n            confirm_exit (boolean): When True create popup to confirm exiting youtube-dl-gui.\n\n            native_hls (boolean): When True youtube-dl will use the native HLS implementation.\n\n            show_completion_popup (boolean): When True youtube-dl-gui will create a popup\n                to inform the user for the download completion.\n\n            confirm_deletion (boolean): When True ask user before item removal.\n\n            nomtime (boolean): When True will not use the Last-modified header to\n                set the file modification time.\n\n            embed_thumbnail (boolean): When True will embed the thumbnail in\n                the audio file as cover art.\n\n            add_metadata (boolean): When True will write metadata to file.\n\n            disable_update (boolean): When True the update process will be disabled.\n\n        '
        self.options = {'save_path': os_path_expanduser('~'), 'save_path_dirs': [os_path_expanduser('~'), os.path.join(os_path_expanduser('~'), 'Downloads'), os.path.join(os_path_expanduser('~'), 'Desktop'), os.path.join(os_path_expanduser('~'), 'Videos'), os.path.join(os_path_expanduser('~'), 'Music')], 'video_format': '0', 'second_video_format': '0', 'to_audio': False, 'keep_video': False, 'audio_format': '', 'audio_quality': '5', 'restrict_filenames': False, 'output_format': 1, 'output_template': os.path.join('%(uploader)s', '%(title)s.%(ext)s'), 'playlist_start': 1, 'playlist_end': 0, 'max_downloads': 0, 'min_filesize': 0, 'max_filesize': 0, 'min_filesize_unit': '', 'max_filesize_unit': '', 'write_subs': False, 'write_all_subs': False, 'write_auto_subs': False, 'embed_subs': False, 'subs_lang': 'en', 'ignore_errors': True, 'open_dl_dir': False, 'write_description': False, 'write_info': False, 'write_thumbnail': False, 'retries': 10, 'user_agent': '', 'referer': '', 'proxy': '', 'shutdown': False, 'sudo_password': '', 'username': '', 'password': '', 'video_password': '', 'youtubedl_path': self.config_path, 'cmd_args': '', 'enable_log': True, 'log_time': True, 'workers_number': 3, 'locale_name': get_default_lang(), 'main_win_size': (740, 490), 'opts_win_size': (640, 490), 'selected_video_formats': ['webm', 'mp4'], 'selected_audio_formats': ['mp3', 'm4a', 'vorbis'], 'selected_format': '0', 'youtube_dl_debug': False, 'ignore_config': True, 'confirm_exit': True, 'native_hls': True, 'show_completion_popup': True, 'confirm_deletion': True, 'nomtime': False, 'embed_thumbnail': False, 'add_metadata': False, 'disable_update': False}
        new_path = '/usr/bin'
        if self.options['disable_update'] and os.name != 'nt' and os_path_exists(new_path):
            self.options['youtubedl_path'] = new_path

    def load_from_file(self):
        if False:
            return 10
        'Load options from settings file. '
        if not os_path_exists(self.settings_file):
            return
        with open(self.settings_file, 'rb') as settings_file:
            try:
                options = json.load(settings_file)
                if self._settings_are_valid(options):
                    self.options = options
            except:
                self.load_default()

    def save_to_file(self):
        if False:
            i = 10
            return i + 15
        'Save options to settings file. '
        check_path(self.config_path)
        with open(self.settings_file, 'wb') as settings_file:
            options = self._get_options()
            json.dump(options, settings_file, indent=4, separators=(',', ': '))

    def _settings_are_valid(self, settings_dictionary):
        if False:
            i = 10
            return i + 15
        'Check settings.json dictionary.\n\n        Args:\n            settings_dictionary (dict): Options dictionary loaded\n                from the settings file. See load_from_file() method.\n\n        Returns:\n            True if settings.json dictionary is valid, else False.\n\n        '
        VALID_VIDEO_FORMAT = ('0', '17', '36', '5', '34', '35', '43', '44', '45', '46', '18', '22', '37', '38', '160', '133', '134', '135', '136', '137', '264', '138', '242', '243', '244', '247', '248', '271', '272', '82', '83', '84', '85', '100', '101', '102', '139', '140', '141', '171', '172')
        VALID_AUDIO_FORMAT = ('mp3', 'wav', 'aac', 'm4a', 'vorbis', 'opus', 'flac', '')
        VALID_AUDIO_QUALITY = ('0', '5', '9')
        VALID_FILESIZE_UNIT = ('', 'k', 'm', 'g', 't', 'p', 'e', 'z', 'y')
        VALID_SUB_LANGUAGE = ('en', 'el', 'pt', 'fr', 'it', 'ru', 'es', 'de', 'he', 'sv', 'tr')
        MIN_FRAME_SIZE = 100
        settings_dictionary['main_win_size'] = decode_tuple(settings_dictionary['main_win_size'])
        settings_dictionary['opts_win_size'] = decode_tuple(settings_dictionary['opts_win_size'])
        for key in self.options:
            if key not in settings_dictionary:
                return False
            if type(self.options[key]) != type(settings_dictionary[key]):
                return False
        rules_dict = {'video_format': FORMATS.keys(), 'second_video_format': VALID_VIDEO_FORMAT, 'audio_format': VALID_AUDIO_FORMAT, 'audio_quality': VALID_AUDIO_QUALITY, 'output_format': OUTPUT_FORMATS.keys(), 'min_filesize_unit': VALID_FILESIZE_UNIT, 'max_filesize_unit': VALID_FILESIZE_UNIT, 'subs_lang': VALID_SUB_LANGUAGE}
        for (key, valid_list) in rules_dict.items():
            if settings_dictionary[key] not in valid_list:
                return False
        if settings_dictionary['workers_number'] < 1:
            return False
        for size in settings_dictionary['main_win_size']:
            if size < MIN_FRAME_SIZE:
                return False
        for size in settings_dictionary['opts_win_size']:
            if size < MIN_FRAME_SIZE:
                return False
        return True

    def _get_options(self):
        if False:
            return 10
        'Return options dictionary without SENSITIVE_KEYS. '
        temp_options = self.options.copy()
        for key in self.SENSITIVE_KEYS:
            temp_options[key] = ''
        temp_options['main_win_size'] = encode_tuple(temp_options['main_win_size'])
        temp_options['opts_win_size'] = encode_tuple(temp_options['opts_win_size'])
        return temp_options