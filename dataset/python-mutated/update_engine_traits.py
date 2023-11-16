"""Update :py:obj:`searx.enginelib.traits.EngineTraitsMap` and :origin:`searx/languages.py`

:py:obj:`searx.enginelib.traits.EngineTraitsMap.ENGINE_TRAITS_FILE`:
  Persistence of engines traits, fetched from the engines.

:origin:`searx/languages.py`
  Is generated  from intersecting each engine's supported traits.

The script :origin:`searxng_extra/update/update_engine_traits.py` is called in
the :origin:`CI Update data ... <.github/workflows/data-update.yml>`

"""
from unicodedata import lookup
from pathlib import Path
from pprint import pformat
import babel
from searx import settings, searx_dir
from searx import network
from searx.engines import load_engines
from searx.enginelib.traits import EngineTraitsMap
languages_file = Path(searx_dir) / 'sxng_locales.py'
languages_file_header = "# -*- coding: utf-8 -*-\n'''List of SearXNG's locale codes.\n\nThis file is generated automatically by::\n\n   ./manage pyenv.cmd searxng_extra/update/update_engine_traits.py\n'''\n\nsxng_locales = (\n"
languages_file_footer = ",\n)\n'''\nA list of five-digit tuples:\n\n0. SearXNG's internal locale tag (a language or region tag)\n1. Name of the language (:py:obj:`babel.core.Locale.get_language_name`)\n2. For region tags the name of the region (:py:obj:`babel.core.Locale.get_territory_name`).\n   Empty string for language tags.\n3. English language name (from :py:obj:`babel.core.Locale.english_name`)\n4. Unicode flag (emoji) that fits to SearXNG's internal region tag. Languages\n   are represented by a globe (🌐)\n\n.. code:: python\n\n   ('en',    'English', '',              'English', '🌐'),\n   ('en-CA', 'English', 'Canada',        'English', '🇨🇦'),\n   ('en-US', 'English', 'United States', 'English', '🇺🇸'),\n   ..\n   ('fr',    'Français', '',             'French',  '🌐'),\n   ('fr-BE', 'Français', 'Belgique',     'French',  '🇧🇪'),\n   ('fr-CA', 'Français', 'Canada',       'French',  '🇨🇦'),\n\n:meta hide-value:\n'''\n"
lang2emoji = {'ha': '🇳🇪', 'bs': '🇧🇦', 'jp': '🇯🇵', 'ua': '🇺🇦', 'he': '🇮🇱'}

def main():
    if False:
        print('Hello World!')
    load_engines(settings['engines'])
    traits_map = fetch_traits_map()
    sxng_tag_list = filter_locales(traits_map)
    write_languages_file(sxng_tag_list)

def fetch_traits_map():
    if False:
        return 10
    'Fetchs supported languages for each engine and writes json file with those.'
    network.set_timeout_for_thread(10.0)

    def log(msg):
        if False:
            i = 10
            return i + 15
        print(msg)
    traits_map = EngineTraitsMap.fetch_traits(log=log)
    print('fetched properties from %s engines' % len(traits_map))
    print('write json file: %s' % traits_map.ENGINE_TRAITS_FILE)
    traits_map.save_data()
    return traits_map

def filter_locales(traits_map: EngineTraitsMap):
    if False:
        return 10
    'Filter language & region tags by a threshold.'
    min_eng_per_region = 11
    min_eng_per_lang = 13
    _ = {}
    for eng in traits_map.values():
        for reg in eng.regions.keys():
            _[reg] = _.get(reg, 0) + 1
    regions = set((k for (k, v) in _.items() if v >= min_eng_per_region))
    lang_from_region = set((k.split('-')[0] for k in regions))
    _ = {}
    for eng in traits_map.values():
        for lang in eng.languages.keys():
            if '_' in lang:
                continue
            _[lang] = _.get(lang, 0) + 1
    languages = set((k for (k, v) in _.items() if v >= min_eng_per_lang))
    sxng_tag_list = set()
    sxng_tag_list.update(regions)
    sxng_tag_list.update(lang_from_region)
    sxng_tag_list.update(languages)
    return sxng_tag_list

def write_languages_file(sxng_tag_list):
    if False:
        return 10
    language_codes = []
    for sxng_tag in sorted(sxng_tag_list):
        sxng_locale: babel.Locale = babel.Locale.parse(sxng_tag, sep='-')
        flag = get_unicode_flag(sxng_locale) or ''
        item = (sxng_tag, sxng_locale.get_language_name().title(), sxng_locale.get_territory_name() or '', sxng_locale.english_name.split(' (')[0], UnicodeEscape(flag))
        language_codes.append(item)
    language_codes = tuple(language_codes)
    with open(languages_file, 'w', encoding='utf-8') as new_file:
        file_content = '{header} {language_codes}{footer}'.format(header=languages_file_header, language_codes=pformat(language_codes, width=120, indent=4)[1:-1], footer=languages_file_footer)
        new_file.write(file_content)
        new_file.close()

class UnicodeEscape(str):
    """Escape unicode string in :py:obj:`pprint.pformat`"""

    def __repr__(self):
        if False:
            while True:
                i = 10
        return "'" + ''.join([chr(c) for c in self.encode('unicode-escape')]) + "'"

def get_unicode_flag(locale: babel.Locale):
    if False:
        i = 10
        return i + 15
    'Determine a unicode flag (emoji) that fits to the ``locale``'
    emoji = lang2emoji.get(locale.language)
    if emoji:
        return emoji
    if not locale.territory:
        return '🌐'
    emoji = lang2emoji.get(locale.territory.lower())
    if emoji:
        return emoji
    try:
        c1 = lookup('REGIONAL INDICATOR SYMBOL LETTER ' + locale.territory[0])
        c2 = lookup('REGIONAL INDICATOR SYMBOL LETTER ' + locale.territory[1])
    except KeyError as exc:
        print('ERROR: %s --> %s' % (locale, exc))
        return None
    return c1 + c2
if __name__ == '__main__':
    main()