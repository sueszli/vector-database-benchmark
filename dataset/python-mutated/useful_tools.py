try:
    from PySide6.QtWidgets import QStyleFactory
except:
    from PyQt5.QtWidgets import QStyleFactory
from persepolis.constants.Os import OS
import urllib.parse
import subprocess
import platform
import sys
import os
try:
    from persepolis.scripts import logger
    logger_availability = True
except:
    logger_availability = False
os_type = platform.system()
home_address = os.path.expanduser('~')

def determineConfigFolder():
    if False:
        print('Hello World!')
    if os_type in OS.UNIX_LIKE:
        config_folder = os.path.join(home_address, '.config/persepolis_download_manager')
    elif os_type == OS.OSX:
        config_folder = os.path.join(home_address, 'Library/Application Support/persepolis_download_manager')
    elif os_type == OS.WINDOWS:
        config_folder = os.path.join(home_address, 'AppData', 'Local', 'persepolis_download_manager')
    return config_folder

def osAndDesktopEnvironment():
    if False:
        while True:
            i = 10
    desktop_env = None
    if os_type in OS.UNIX_LIKE:
        desktop_env = os.environ.get('XDG_CURRENT_DESKTOP')
    return (os_type, desktop_env)

def humanReadableSize(size, input_type='file_size'):
    if False:
        print('Hello World!')
    labels = ['KiB', 'MiB', 'GiB', 'TiB']
    i = -1
    if size < 1024:
        return str(size) + ' B'
    while size >= 1024:
        i += 1
        size = size / 1024
    if input_type == 'speed':
        j = 0
    else:
        j = 1
    if i > j:
        return str(round(size, 2)) + ' ' + labels[i]
    else:
        return str(round(size, None)) + ' ' + labels[i]

def convertToByte(file_size):
    if False:
        for i in range(10):
            print('nop')
    if file_size[-2:] != ' B':
        unit = file_size[-3:]
        if unit == 'GiB' or unit == 'TiB':
            size_value = float(file_size[:-4])
        else:
            size_value = int(float(file_size[:-4]))
    else:
        unit = None
        size_value = int(float(file_size[:-3]))
    if not unit:
        in_byte_value = size_value
    elif unit == 'KiB':
        in_byte_value = size_value * 1024
    elif unit == 'MiB':
        in_byte_value = size_value * 1024 * 1024
    elif unit == 'GiB':
        in_byte_value = size_value * 1024 * 1024 * 1024
    elif unit == 'TiB':
        in_byte_value = size_value * 1024 * 1024 * 1024 * 1024
    return int(in_byte_value)

def freeSpace(dir):
    if False:
        while True:
            i = 10
    try:
        import psutil
    except:
        if logger_availability:
            logger.sendToLog('psutil in not installed!', 'ERROR')
        return None
    try:
        dir_space = psutil.disk_usage(dir)
        free_space = dir_space.free
        return int(free_space)
    except Exception as e:
        if logger_availability:
            logger.sendToLog("persepolis couldn't find free space value:\n" + str(e), 'ERROR')
        return None

def returnDefaultSettings():
    if False:
        return 10
    (os_type, desktop_env) = osAndDesktopEnvironment()
    if os_type != OS.WINDOWS:
        download_path_temp = home_address + '/.persepolis'
    else:
        download_path_temp = os.path.join(home_address, 'AppData', 'Local', 'persepolis')
    download_path = os.path.join(home_address, 'Downloads', 'Persepolis')
    available_styles = QStyleFactory.keys()
    style = 'Fusion'
    color_scheme = 'Dark Fusion'
    icons = 'Breeze-Dark'
    if os_type in OS.UNIX_LIKE:
        if desktop_env == 'KDE':
            if 'Breeze' in available_styles:
                style = 'Breeze'
                color_scheme = 'System'
        else:
            gtk3_confing_file_path = os.path.join(home_address, '.config', 'gtk-3.0', 'settings.ini')
            if not os.path.isfile(gtk3_confing_file_path):
                if os.path.isfile('/etc/gtk-3.0/settings.ini'):
                    gtk3_confing_file_path = '/etc/gtk-3.0/settings.ini'
                else:
                    gtk3_confing_file_path = None
            dark_theme = False
            if gtk3_confing_file_path:
                with open(gtk3_confing_file_path) as f:
                    for line in f:
                        if 'gtk-application-prefer-dark-theme' in line:
                            if 'true' in line:
                                dark_theme = True
                            else:
                                dark_theme = False
            if dark_theme:
                icons = 'Breeze-Dark'
                if 'Adwaita-Dark' in available_styles:
                    style = 'Adwaita-Dark'
                    color_scheme = 'System'
            else:
                icons = 'Breeze'
                if 'Adwaita' in available_styles:
                    style = 'Adwaita'
                    color_scheme = 'System'
                else:
                    style = 'Fusion'
                    color_scheme = 'Light Fusion'
    elif os_type == OS.OSX:
        if 'macintosh' in available_styles:
            style = 'macintosh'
            color_scheme = 'System'
            icons = 'Breeze'
    elif os_type == OS.WINDOWS:
        style = 'Fusion'
        color_scheme = 'Dark Fusion'
        icons = 'Breeze-Dark'
    else:
        style = 'Fusion'
        color_scheme = 'Dark Fusion'
        icons = 'Breeze-Dark'
    delete_shortcut = 'Ctrl+D'
    remove_shortcut = 'Ctrl+R'
    add_new_download_shortcut = 'Ctrl+N'
    import_text_shortcut = 'Ctrl+O'
    video_finder_shortcut = 'Ctrl+V'
    quit_shortcut = 'Ctrl+Q'
    hide_window_shortcut = 'Ctrl+W'
    move_up_selection_shortcut = 'Ctrl+Up'
    move_down_selection_shortcut = 'Ctrl+Down'
    default_setting_dict = {'locale': 'en_US', 'toolbar_icon_size': 32, 'wait-queue': [0, 0], 'awake': 'no', 'custom-font': 'no', 'column0': 'yes', 'column1': 'yes', 'column2': 'yes', 'column3': 'yes', 'column4': 'yes', 'column5': 'yes', 'column6': 'yes', 'column7': 'yes', 'column10': 'yes', 'column11': 'yes', 'column12': 'yes', 'subfolder': 'yes', 'startup': 'no', 'show-progress': 'yes', 'show-menubar': 'no', 'show-sidepanel': 'yes', 'rpc-port': 6801, 'notification': 'Native notification', 'after-dialog': 'yes', 'tray-icon': 'yes', 'browser-persepolis': 'yes', 'hide-window': 'yes', 'max-tries': 5, 'retry-wait': 0, 'timeout': 60, 'connections': 16, 'download_path_temp': download_path_temp, 'download_path': download_path, 'sound': 'yes', 'sound-volume': 100, 'style': style, 'color-scheme': color_scheme, 'icons': icons, 'font': 'Ubuntu', 'font-size': 9, 'aria2_path': '', 'video_finder/max_links': '3', 'shortcuts/delete_shortcut': delete_shortcut, 'shortcuts/remove_shortcut': remove_shortcut, 'shortcuts/add_new_download_shortcut': add_new_download_shortcut, 'shortcuts/import_text_shortcut': import_text_shortcut, 'shortcuts/video_finder_shortcut': video_finder_shortcut, 'shortcuts/quit_shortcut': quit_shortcut, 'shortcuts/hide_window_shortcut': hide_window_shortcut, 'shortcuts/move_up_selection_shortcut': move_up_selection_shortcut, 'shortcuts/move_down_selection_shortcut': move_down_selection_shortcut, 'dont-check-certificate': 'no'}
    return default_setting_dict

def muxer(parent, video_finder_dictionary):
    if False:
        for i in range(10):
            print('nop')
    result_dictionary = {'error': 'no_error', 'ffmpeg_error_message': None, 'final_path': None, 'final_size': None}
    video_file_dictionary = parent.persepolis_db.searchGidInAddLinkTable(video_finder_dictionary['video_gid'])
    audio_file_dictionary = parent.persepolis_db.searchGidInAddLinkTable(video_finder_dictionary['audio_gid'])
    video_file_path = video_file_dictionary['download_path']
    audio_file_path = audio_file_dictionary['download_path']
    final_path = video_finder_dictionary['download_path']
    video_file_size = parent.persepolis_db.searchGidInDownloadTable(video_finder_dictionary['video_gid'])['size']
    audio_file_size = parent.persepolis_db.searchGidInDownloadTable(video_finder_dictionary['audio_gid'])['size']
    video_file_size = convertToByte(video_file_size)
    audio_file_size = convertToByte(audio_file_size)
    final_file_size = video_file_size + audio_file_size
    free_space = freeSpace(final_path)
    if free_space:
        if final_file_size > free_space:
            result_dictionary['error'] = 'not enough free space'
        else:
            final_file_name = urllib.parse.unquote(os.path.basename(video_file_path))
            file_name_split = final_file_name.split('.')
            video_extension = file_name_split[-1]
            if video_extension == 'webm':
                extension_length = len(file_name_split[-1]) + 1
                final_file_name = final_file_name[0:-extension_length] + '.mkv'
            if parent.persepolis_setting.value('settings/download_path') == final_path:
                if parent.persepolis_setting.value('settings/subfolder') == 'yes':
                    final_path = os.path.join(final_path, 'Videos')
            i = 1
            final_path_plus_name = os.path.join(final_path, final_file_name)
            while os.path.isfile(final_path_plus_name):
                extension_length = len(file_name_split[-1]) + 1
                new_name = final_file_name[0:-extension_length] + '_' + str(i) + final_file_name[-extension_length:]
                final_path_plus_name = os.path.join(final_path, new_name)
                i = i + 1
            if os_type in OS.UNIX_LIKE:
                pipe = subprocess.Popen(['ffmpeg', '-i', video_file_path, '-i', audio_file_path, '-c', 'copy', '-shortest', '-map', '0:v:0', '-map', '1:a:0', '-loglevel', 'error', '-strict', '-2', final_path_plus_name], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
            elif os_type == OS.DARWIN:
                cwd = sys.argv[0]
                current_directory = os.path.dirname(cwd)
                ffmpeg_path = os.path.join(current_directory, 'ffmpeg')
                pipe = subprocess.Popen([ffmpeg_path, '-i', video_file_path, '-i', audio_file_path, '-c', 'copy', '-shortest', '-map', '0:v:0', '-map', '1:a:0', '-loglevel', 'error', '-strict', '-2', final_path_plus_name], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
            elif os_type == OS.WINDOWS:
                cwd = sys.argv[0]
                current_directory = os.path.dirname(cwd)
                ffmpeg_path = os.path.join(current_directory, 'ffmpeg.exe')
                NO_WINDOW = 134217728
                pipe = subprocess.Popen([ffmpeg_path, '-i', video_file_path, '-i', audio_file_path, '-c', 'copy', '-shortest', '-map', '0:v:0', '-map', '1:a:0', '-loglevel', 'error', '-strict', '-2', final_path_plus_name], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, creationflags=NO_WINDOW)
            if pipe.wait() == 0:
                result_dictionary['error'] = 'no error'
                result_dictionary['final_path'] = final_path_plus_name
                result_dictionary['final_size'] = humanReadableSize(final_file_size)
            else:
                result_dictionary['error'] = 'ffmpeg error'
                (out, ffmpeg_error_message) = pipe.communicate()
                result_dictionary['ffmpeg_error_message'] = ffmpeg_error_message.decode('utf-8', 'ignore')
    return result_dictionary