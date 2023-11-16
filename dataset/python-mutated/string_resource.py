"""
Module for reading plaintext-based language files.
"""
from __future__ import annotations
import typing
import re
from ....log import dbg
from ...entity_object.conversion.stringresource import StringResource
from ...value_object.read.media.langcodes import LANGCODES_DE1, LANGCODES_DE2, LANGCODES_HD
from ...value_object.read.media.pefile import PEFile
from ...value_object.read.media_types import MediaType
if typing.TYPE_CHECKING:
    from argparse import Namespace
    from openage.util.fslike.directory import Directory
    from openage.util.fslike.path import Path
    from openage.util.fslike.wrapper import GuardedFile

def get_string_resources(args: Namespace) -> StringResource:
    if False:
        return 10
    ' reads the (language) string resources '
    stringres = StringResource()
    srcdir = args.srcdir
    game_edition = args.game_version.edition
    language_files = game_edition.media_paths[MediaType.LANGUAGE]
    for language_file in language_files:
        if game_edition.game_id in ('ROR', 'AOC', 'SWGB'):
            pefile = PEFile(srcdir[language_file].open('rb'))
            stringres.fill_from(pefile.resources().strings)
        elif game_edition.game_id == 'HDEDITION':
            strings = read_hd_language_file(srcdir, language_file)
            stringres.fill_from(strings)
        elif game_edition.game_id == 'AOE1DE':
            strings = read_de1_language_file(srcdir, language_file)
            stringres.fill_from(strings)
        elif game_edition.game_id == 'AOE2DE':
            strings = read_de2_language_file(srcdir, language_file)
            stringres.fill_from(strings)
        else:
            raise KeyError(f'No service found for parsing language files of version {game_edition.game_id}')
    return stringres

def read_age2_hd_fe_stringresources(stringres: StringResource, path: Path) -> int:
    if False:
        while True:
            i = 10
    '\n    Fill the string resources from text specifications found\n    in the given path.\n\n    In age2hd forgotten those are stored in plain text files.\n\n    The data is stored in the `stringres` storage.\n    '
    count = 0
    for lang in path.list():
        try:
            if lang == b'_common':
                continue
            if lang == b'_packages':
                continue
            if lang.lower() == b'.ds_store'.lower():
                continue
            langfilename = [lang.decode(), 'strings', 'key-value', 'key-value-strings-utf8.txt']
            with path[langfilename].open('rb') as langfile:
                stringres.fill_from(read_hd_language_file_old(langfile, lang))
            count += 1
        except FileNotFoundError:
            pass
    return count

def read_age2_hd_3x_stringresources(stringres: StringResource, srcdir: Directory) -> int:
    if False:
        i = 10
        return i + 15
    '\n    HD Edition 3.x and below store language .txt files\n    in the Bin/ folder.\n    Specific language strings are in Bin/$LANG/*.txt.\n\n    The data is stored in the `stringres` storage.\n    '
    count = 0
    for lang in srcdir['bin'].list():
        lang_path = srcdir['bin', lang.decode()]
        if not lang_path.is_dir():
            continue
        if lang_path['language.dll'].is_file():
            for name in ['language.dll', 'language_x1.dll', 'language_x1_p1.dll']:
                pefile = PEFile(lang_path[name].open('rb'))
                stringres.fill_from(pefile.resources().strings)
                count += 1
        else:
            for basename in lang_path.list():
                with lang_path[basename].open('rb') as langfile:
                    stringres.fill_from(read_hd_language_file_old(langfile, lang, enc='iso-8859-1'))
                count += 1
    return count

def read_hd_language_file_old(fileobj: GuardedFile, langcode: str, enc: str='utf-8') -> dict[str, StringResource]:
    if False:
        print('Hello World!')
    "\n    Takes a file object, and the file's language code.\n    "
    dbg('parse HD Language file %s', langcode)
    strings = {}
    for line in fileobj.read().decode(enc).split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        (string_id, string) = line.split(None, 1)
        strings[string_id] = string
    fileobj.close()
    lang = LANGCODES_HD.get(langcode, langcode)
    return {lang: strings}

def read_hd_language_file(srcdir: Directory, language_file: GuardedFile, enc: str='utf-8') -> dict[str, StringResource]:
    if False:
        return 10
    '\n    HD Edition stores language .txt files in the resources/ folder.\n    Specific language strings are in resources/$LANG/strings/key-value/*.txt.\n\n    The data is stored in the `stringres` storage.\n    '
    langcode = language_file.split('/')[1]
    dbg('parse HD Language file %s', langcode)
    strings = {}
    fileobj = srcdir[language_file].open('rb')
    for line in fileobj.read().decode(enc).split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        (string_id, string) = line.split(None, 1)
        strings[string_id] = string
    fileobj.close()
    lang = LANGCODES_HD.get(langcode, langcode)
    return {lang: strings}

def read_de1_language_file(srcdir: Directory, language_file: GuardedFile) -> dict[str, StringResource]:
    if False:
        return 10
    '\n    Definitve Edition stores language .txt files in the Localization folder.\n    Specific language strings are in Data/Localization/$LANG/strings.txt.\n\n    The data is stored in the `stringres` storage.\n    '
    langcode = language_file.split('/')[2]
    dbg('parse DE1 Language file %s', langcode)
    strings = {}
    fileobj = srcdir[language_file].open('rb')
    for line in fileobj.read().decode('utf-8').split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        (string_id, string) = re.split(',|\\s', line, maxsplit=1)
        strings[string_id] = string
    fileobj.close()
    lang = LANGCODES_DE1.get(langcode, langcode)
    return {lang: strings}

def read_de2_language_file(srcdir: Directory, language_file: GuardedFile) -> dict[str, StringResource]:
    if False:
        return 10
    '\n    Definitve Edition stores language .txt files in the resources/ folder.\n    Specific language strings are in resources/$LANG/strings/key-value/*.txt.\n\n    The data is stored in the `stringres` storage.\n    '
    langcode = language_file.split('/')[1]
    dbg('parse DE2 Language file %s', langcode)
    strings = {}
    fileobj = srcdir[language_file].open('rb')
    for line in fileobj.read().decode('utf-8').split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        (string_id, string) = line.split(None, 1)
        strings[string_id] = string
    fileobj.close()
    lang = LANGCODES_DE2.get(langcode, langcode)
    return {lang: strings}