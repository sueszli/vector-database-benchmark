import re
from picard.config import get_config
from picard.const.sys import IS_WIN
from picard.metadata import Metadata
from picard.script import ScriptParser
from picard.util import replace_win32_incompat, sanitize_filename
from picard.util.textencoding import replace_non_ascii
_re_replace_underscores = re.compile('[\\s_]+')

def script_to_filename_with_metadata(naming_format, metadata, file=None, settings=None):
    if False:
        print('Hello World!')
    'Creates a valid filename from a script with the given metadata.\n\n    Args:\n        naming_format: A string containing the tagger script. The result of\n            executing this script will be the filename.\n        metadata: A Metadata object. The metadata will not be modified.\n        file: A File object (optional)\n        settings: The settings. If not set config.setting will be used.\n\n    Returns:\n        A tuple with the filename as first element and the updated metadata\n        with changes from the script as second.\n    '
    if settings is None:
        config = get_config()
        settings = config.setting
    win_compat = IS_WIN or settings['windows_compatibility']
    new_metadata = Metadata()
    replace_dir_separator = settings['replace_dir_separator']
    for name in metadata:
        new_metadata[name] = [sanitize_filename(str(v), repl=replace_dir_separator, win_compat=win_compat) for v in metadata.getall(name)]
    naming_format = naming_format.replace('\t', '').replace('\n', '')
    filename = ScriptParser().eval(naming_format, new_metadata, file)
    if settings['ascii_filenames']:
        filename = replace_non_ascii(filename, pathsave=True, win_compat=win_compat)
    if win_compat:
        filename = replace_win32_incompat(filename, replacements=settings['win_compat_replacements'])
    if settings['replace_spaces_with_underscores']:
        filename = _re_replace_underscores.sub('_', filename.strip())
    filename = filename.replace('\x00', '')
    return (filename, new_metadata)

def script_to_filename(naming_format, metadata, file=None, settings=None):
    if False:
        print('Hello World!')
    'Creates a valid filename from a script with the given metadata.\n\n    Args:\n        naming_format: A string containing the tagger script. The result of\n            executing this script will be the filename.\n        metadata: A Metadata object. The metadata will not be modified.\n        file: A File object (optional)\n        settings: The settings. If not set config.setting will be used.\n\n    Returns:\n        The filename.\n    '
    (filename, _unused) = script_to_filename_with_metadata(naming_format, metadata, file, settings)
    return filename