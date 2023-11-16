"""
This module exposes various constants and helpers useful for decky plugins.

* Plugin's settings and configurations should be stored under `DECKY_PLUGIN_SETTINGS_DIR`.
* Plugin's runtime data should be stored under `DECKY_PLUGIN_RUNTIME_DIR`.
* Plugin's persistent log files should be stored under `DECKY_PLUGIN_LOG_DIR`.

Avoid writing outside of `DECKY_HOME`, storing under the suggested paths is strongly recommended.

Some basic migration helpers are available: `migrate_any`, `migrate_settings`, `migrate_runtime`, `migrate_logs`.

A logging facility `logger` is available which writes to the recommended location.
"""
__version__ = '0.1.0'
import os
import subprocess
import logging
import time
'\nConstants\n'
HOME: str = os.getenv('HOME', default='')
"\nThe home directory of the effective user running the process.\nEnvironment variable: `HOME`.\nIf `root` was specified in the plugin's flags it will be `/root` otherwise the user whose home decky resides in.\ne.g.: `/home/deck`\n"
USER: str = os.getenv('USER', default='')
"\nThe effective username running the process.\nEnvironment variable: `USER`.\nIt would be `root` if `root` was specified in the plugin's flags otherwise the user whose home decky resides in.\ne.g.: `deck`\n"
DECKY_VERSION: str = os.getenv('DECKY_VERSION', default='')
'\nThe version of the decky loader.\nEnvironment variable: `DECKY_VERSION`.\ne.g.: `v2.5.0-pre1`\n'
DECKY_USER: str = os.getenv('DECKY_USER', default='')
'\nThe user whose home decky resides in.\nEnvironment variable: `DECKY_USER`.\ne.g.: `deck`\n'
DECKY_USER_HOME: str = os.getenv('DECKY_USER_HOME', default='')
'\nThe home of the user where decky resides in.\nEnvironment variable: `DECKY_USER_HOME`.\ne.g.: `/home/deck`\n'
DECKY_HOME: str = os.getenv('DECKY_HOME', default='')
'\nThe root of the decky folder.\nEnvironment variable: `DECKY_HOME`.\ne.g.: `/home/deck/homebrew`\n'
DECKY_PLUGIN_SETTINGS_DIR: str = os.getenv('DECKY_PLUGIN_SETTINGS_DIR', default='')
'\nThe recommended path in which to store configuration files (created automatically).\nEnvironment variable: `DECKY_PLUGIN_SETTINGS_DIR`.\ne.g.: `/home/deck/homebrew/settings/decky-plugin-template`\n'
DECKY_PLUGIN_RUNTIME_DIR: str = os.getenv('DECKY_PLUGIN_RUNTIME_DIR', default='')
'\nThe recommended path in which to store runtime data (created automatically).\nEnvironment variable: `DECKY_PLUGIN_RUNTIME_DIR`.\ne.g.: `/home/deck/homebrew/data/decky-plugin-template`\n'
DECKY_PLUGIN_LOG_DIR: str = os.getenv('DECKY_PLUGIN_LOG_DIR', default='')
'\nThe recommended path in which to store persistent logs (created automatically).\nEnvironment variable: `DECKY_PLUGIN_LOG_DIR`.\ne.g.: `/home/deck/homebrew/logs/decky-plugin-template`\n'
DECKY_PLUGIN_DIR: str = os.getenv('DECKY_PLUGIN_DIR', default='')
"\nThe root of the plugin's directory.\nEnvironment variable: `DECKY_PLUGIN_DIR`.\ne.g.: `/home/deck/homebrew/plugins/decky-plugin-template`\n"
DECKY_PLUGIN_NAME: str = os.getenv('DECKY_PLUGIN_NAME', default='')
"\nThe name of the plugin as specified in the 'plugin.json'.\nEnvironment variable: `DECKY_PLUGIN_NAME`.\ne.g.: `Example Plugin`\n"
DECKY_PLUGIN_VERSION: str = os.getenv('DECKY_PLUGIN_VERSION', default='')
"\nThe version of the plugin as specified in the 'package.json'.\nEnvironment variable: `DECKY_PLUGIN_VERSION`.\ne.g.: `0.0.1`\n"
DECKY_PLUGIN_AUTHOR: str = os.getenv('DECKY_PLUGIN_AUTHOR', default='')
"\nThe author of the plugin as specified in the 'plugin.json'.\nEnvironment variable: `DECKY_PLUGIN_AUTHOR`.\ne.g.: `John Doe`\n"
__cur_time = time.strftime('%Y-%m-%d %H.%M.%S')
DECKY_PLUGIN_LOG: str = os.path.join(DECKY_PLUGIN_LOG_DIR, f'{__cur_time}.log')
"\nThe path to the plugin's main logfile.\nEnvironment variable: `DECKY_PLUGIN_LOG`.\ne.g.: `/home/deck/homebrew/logs/decky-plugin-template/plugin.log`\n"
'\nMigration helpers\n'

def migrate_any(target_dir: str, *files_or_directories: str) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    '\n    Migrate files and directories to a new location and remove old locations.\n    Specified files will be migrated to `target_dir`.\n    Specified directories will have their contents recursively migrated to `target_dir`.\n\n    Returns the mapping of old -> new location.\n    '
    file_map: dict[str, str] = {}
    for f in files_or_directories:
        if not os.path.exists(f):
            file_map[f] = ''
            continue
        if os.path.isdir(f):
            src_dir = f
            src_file = '.'
            file_map[f] = target_dir
        else:
            src_dir = os.path.dirname(f)
            src_file = os.path.basename(f)
            file_map[f] = os.path.join(target_dir, src_file)
        subprocess.run(['sh', '-c', 'mkdir -p "$3"; tar -cf - -C "$1" "$2" | tar -xf - -C "$3" && rm -rf "$4"', '_', src_dir, src_file, target_dir, f])
    return file_map

def migrate_settings(*files_or_directories: str) -> dict[str, str]:
    if False:
        return 10
    '\n    Migrate files and directories relating to plugin settings to the recommended location and remove old locations.\n    Specified files will be migrated to `DECKY_PLUGIN_SETTINGS_DIR`.\n    Specified directories will have their contents recursively migrated to `DECKY_PLUGIN_SETTINGS_DIR`.\n\n    Returns the mapping of old -> new location.\n    '
    return migrate_any(DECKY_PLUGIN_SETTINGS_DIR, *files_or_directories)

def migrate_runtime(*files_or_directories: str) -> dict[str, str]:
    if False:
        while True:
            i = 10
    '\n    Migrate files and directories relating to plugin runtime data to the recommended location and remove old locations\n    Specified files will be migrated to `DECKY_PLUGIN_RUNTIME_DIR`.\n    Specified directories will have their contents recursively migrated to `DECKY_PLUGIN_RUNTIME_DIR`.\n\n    Returns the mapping of old -> new location.\n    '
    return migrate_any(DECKY_PLUGIN_RUNTIME_DIR, *files_or_directories)

def migrate_logs(*files_or_directories: str) -> dict[str, str]:
    if False:
        return 10
    '\n    Migrate files and directories relating to plugin logs to the recommended location and remove old locations.\n    Specified files will be migrated to `DECKY_PLUGIN_LOG_DIR`.\n    Specified directories will have their contents recursively migrated to `DECKY_PLUGIN_LOG_DIR`.\n\n    Returns the mapping of old -> new location.\n    '
    return migrate_any(DECKY_PLUGIN_LOG_DIR, *files_or_directories)
'\nLogging\n'
try:
    for x in [entry.name for entry in sorted(os.scandir(DECKY_PLUGIN_LOG_DIR), key=lambda x: x.stat().st_mtime, reverse=True) if entry.name.endswith('.log')][4:]:
        os.unlink(os.path.join(DECKY_PLUGIN_LOG_DIR, x))
except Exception as e:
    print(f'Failed to delete old logs: {str(e)}')
logging.basicConfig(filename=DECKY_PLUGIN_LOG, format='[%(asctime)s][%(levelname)s]: %(message)s', force=True)
logger: logging.Logger = logging.getLogger()
'The main plugin logger writing to `DECKY_PLUGIN_LOG`.'
logger.setLevel(logging.INFO)