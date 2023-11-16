"""
Acquire the sourcedir for the game that is supposed to be converted.
"""
from __future__ import annotations
import platform
import typing
from configparser import ConfigParser
import os
from pathlib import Path
import subprocess
import sys
from typing import AnyStr, Generator
import shutil
import tempfile
from urllib.request import urlopen
from ....log import warn, info, dbg
from ....util.files import which
from ....util.fslike.directory import CaseIgnoringDirectory, Directory
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameEdition
STANDARD_PATH_IN_32BIT_WINEPREFIX = 'drive_c/Program Files/Microsoft Games/Age of Empires II/'
STANDARD_PATH_IN_64BIT_WINEPREFIX = 'drive_c/Program Files (x86)/Microsoft Games/Age of Empires II/'
STANDARD_PATH_IN_WINEPREFIX_STEAM = 'drive_c/Program Files (x86)/Steam/steamapps/common/Age2HD/'
REGISTRY_KEY = 'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Microsoft Games\\'
REGISTRY_SUFFIX_AOK = 'Age of Empires\\2.0'
REGISTRY_SUFFIX_TC = 'Age of Empires II: The Conquerors Expansion\\1.0'
TRIAL_URL = 'https://archive.org/download/AgeOfEmpiresIiTheConquerorsDemo/Age2XTrial.exe'

def expand_relative_path(path: str) -> AnyStr:
    if False:
        i = 10
        return i + 15
    'Expand relative path to an absolute one, including abbreviations like\n    ~ and environment variables'
    return os.path.realpath(os.path.expandvars(os.path.expanduser(path)))

def wanna_convert() -> bool:
    if False:
        while True:
            i = 10
    '\n    Ask the user if assets should be converted.\n    '
    answer = None
    while answer is None:
        print('  Do you want to convert assets? [Y/n]')
        user_selection = input('> ')
        if user_selection.lower() in {'yes', 'y', ''}:
            answer = True
        elif user_selection.lower() in {'no', 'n'}:
            answer = False
    return answer

def wanna_download_trial() -> bool:
    if False:
        while True:
            i = 10
    '\n    Ask the user if the AoC trial should be downloaded.\n    '
    answer = None
    while answer is None:
        print('  Do you want to download the AoC trial version? [Y/n]')
        user_selection = input('> ')
        if user_selection.lower() in {'yes', 'y', ''}:
            answer = True
        elif user_selection.lower() in {'no', 'n'}:
            answer = False
    return answer

def wanna_use_wine() -> bool:
    if False:
        print('Hello World!')
    '\n    Ask the user if wine should be used.\n    Wine is not used if user has no wine installed.\n    '
    if not which('wine'):
        return False
    answer = None
    long_prompt = True
    while answer is None:
        if long_prompt:
            print('  Should we call wine to determine an AOE installation? [Y/n]')
            long_prompt = False
        else:
            print("  Don't know what you want. Use wine? [Y/n]")
        user_selection = input('> ')
        if user_selection.lower() in {'yes', 'y', ''}:
            answer = True
        elif user_selection.lower() in {'no', 'n'}:
            answer = False
    return answer

def set_custom_wineprefix() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Allow the customization of the WINEPREFIX environment variable.\n    '
    print("The WINEPREFIX is a separate 'container' for windows software installations.")
    current_wineprefix = os.environ.get('WINEPREFIX')
    if current_wineprefix:
        print(f"Currently: WINEPREFIX='{current_wineprefix}'")
    print('Enter a custom value or leave empty to keep it as-is:')
    while True:
        new_wineprefix = input('WINEPREFIX=')
        if not new_wineprefix:
            break
        new_wineprefix = expand_relative_path(new_wineprefix)
        if (Path(new_wineprefix) / 'drive_c').is_dir():
            break
        print('This does not appear to be a valid WINEPREFIX.')
        print('Enter a valid one, or leave it empty to skip.')
    if new_wineprefix:
        os.environ['WINEPREFIX'] = new_wineprefix

def query_source_dir(proposals: set[str]) -> AnyStr:
    if False:
        i = 10
        return i + 15
    '\n    Query interactively for a conversion source directory.\n    Lists proposals and allows selection if some were found.\n    '
    if proposals:
        print('\nPlease select an Age of Empires installation directory.')
        print('Insert the index of one of the proposals, or any path:')
        proposals = sorted(proposals)
        for (index, proposal) in enumerate(proposals):
            print(f'({index}) {proposal}')
    else:
        print('Could not find any installation directory automatically.')
        print('Please enter an AOE2 install path manually.')
    while True:
        user_selection = input('> ')
        if user_selection.isdecimal() and int(user_selection) < len(proposals):
            sourcedir = proposals[int(user_selection)]
        else:
            sourcedir = user_selection
        sourcedir = expand_relative_path(sourcedir)
        if Path(sourcedir).is_dir():
            break
        warn('No valid existing directory: %s', sourcedir)
    return sourcedir

def acquire_conversion_source_dir(avail_game_eds: list[GameEdition], prev_srcdir_paths: set[str]=None) -> Path:
    if False:
        for i in range(10):
            print('nop')
    '\n    Acquires source dir for the asset conversion.\n\n    Returns a file system-like object that holds all the required files.\n    '
    try:
        proposals = set()
        if prev_srcdir_paths:
            for prev_srcdir_path in prev_srcdir_paths:
                if Path(prev_srcdir_path).is_dir():
                    proposals.add(prev_srcdir_path)
        current_platform = platform.system()
        for game_edition in avail_game_eds:
            install_paths = game_edition.install_paths
            candidates = []
            if current_platform == 'Linux' and 'linux' in install_paths:
                candidates = install_paths['linux']
            elif current_platform == 'Darwin' and 'macos' in install_paths:
                candidates = install_paths['macos']
            elif current_platform == 'Windows' and 'windows' in install_paths:
                candidates = install_paths['windows']
            else:
                continue
            for candidate in candidates:
                if Path(expand_relative_path(candidate)).is_dir():
                    proposals.add(candidate)
        use_trial = False
        if len(proposals) == 0:
            print('\nopenage requires a local game installation for conversion')
            print('but no local installation could be found automatically.')
            use_trial = wanna_download_trial()
        if use_trial:
            sourcedir = download_trial()
        else:
            sourcedir = query_source_dir(proposals)
    except KeyboardInterrupt:
        print('\nInterrupted, aborting')
        sys.exit(0)
    except EOFError:
        print('\nEOF, aborting')
        sys.exit(0)
    print(f"converting from '{sourcedir}'")
    return CaseIgnoringDirectory(sourcedir).root

def download_trial() -> AnyStr:
    if False:
        return 10
    '\n    Download and extract the AoC trial version.\n\n    Does not work yet. TODO: Find an exe unpack solution that works on all platforms\n    '
    print(f'Downloading AoC trial version from {TRIAL_URL}')
    tempdir = tempfile.mkdtemp()
    with urlopen(TRIAL_URL) as response:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            from ....cabextract.cab import CABFile
            cab = CABFile(tmp_file, 415352)
            sourcedir = Directory(tempdir).root
            print(f'Extracting game files to {sourcedir}...')
            dirs = [cab.root]
            while len(dirs) > 0:
                cur_src_dir = dirs[0]
                cur_tgt_dir = sourcedir
                for part in cur_src_dir.parts:
                    cur_tgt_dir = cur_tgt_dir[part]
                cur_tgt_dir.mkdirs()
                dirs.remove(cur_src_dir)
                for path in cur_src_dir.iterdir():
                    if path.is_dir():
                        dirs.append(path)
                    if path.is_file():
                        with cur_tgt_dir[path.name].open('wb') as target_file:
                            with path.open('rb') as source_file:
                                target_file.write(source_file.read())
    return tempdir

def wine_to_real_path(path: str) -> str:
    if False:
        print('Hello World!')
    '\n    Turn a Wine file path (C:\\xyz) into a local filesystem path (~/.wine/xyz)\n    '
    return subprocess.check_output(('winepath', path)).strip().decode()

def unescape_winereg(value: str):
    if False:
        for i in range(10):
            print('nop')
    'Remove quotes and escapes from a Wine registry value'
    return value.strip('"').replace('\\\\\\\\', '\\')

def wine_srcdir_proposals() -> Generator[str, None, None]:
    if False:
        print('Hello World!')
    'Yield a list of directory names where an installation might be found'
    if 'WINEPREFIX' in os.environ:
        yield ('$WINEPREFIX/' + STANDARD_PATH_IN_32BIT_WINEPREFIX)
        yield ('$WINEPREFIX/' + STANDARD_PATH_IN_64BIT_WINEPREFIX)
        yield ('$WINEPREFIX/' + STANDARD_PATH_IN_WINEPREFIX_STEAM)
    yield ('~/.wine/' + STANDARD_PATH_IN_32BIT_WINEPREFIX)
    yield ('~/.wine/' + STANDARD_PATH_IN_64BIT_WINEPREFIX)
    yield ('~/.wine/' + STANDARD_PATH_IN_WINEPREFIX_STEAM)
    try:
        info('using the wine registry to query an installation location...')
        with tempfile.NamedTemporaryFile(mode='rb') as reg_file:
            if not subprocess.call(('wine', 'regedit', '/E', reg_file.name, REGISTRY_KEY)):
                reg_raw_data = reg_file.read()
                try:
                    reg_data = reg_raw_data.decode('utf-16')
                except UnicodeDecodeError:
                    reg_data = reg_raw_data.decode('utf-8', errors='replace')
                lines = reg_data.splitlines()
                del lines[0:2]
                reg_parser = ConfigParser()
                reg_parser.read_string(''.join(lines))
                for suffix in (REGISTRY_SUFFIX_AOK, REGISTRY_SUFFIX_TC):
                    reg_key = REGISTRY_KEY + suffix
                    if reg_key in reg_parser:
                        if '"InstallationDirectory"' in reg_parser[reg_key]:
                            yield wine_to_real_path(unescape_winereg(reg_parser[reg_key]['"InstallationDirectory"']))
                        if '"EXE Path"' in reg_parser[reg_key]:
                            yield wine_to_real_path(unescape_winereg(reg_parser[reg_key]['"EXE Path"']))
    except OSError as error:
        dbg('wine registry extraction failed: %s', error)