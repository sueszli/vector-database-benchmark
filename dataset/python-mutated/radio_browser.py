"""Search radio stations from RadioBrowser by `Advanced station search API`_.

.. _Advanced station search API:
   https://de1.api.radio-browser.info/#Advanced_station_search

"""
from urllib.parse import urlencode
import babel
from flask_babel import gettext
from searx.network import get
from searx.enginelib.traits import EngineTraits
from searx.locales import language_tag
traits: EngineTraits
about = {'website': 'https://www.radio-browser.info/', 'wikidata_id': 'Q111664849', 'official_api_documentation': 'https://de1.api.radio-browser.info/', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
paging = True
categories = ['music', 'radio']
base_url = 'https://de1.api.radio-browser.info'
number_of_results = 10
station_filters = []
'A list of filters to be applied to the search of radio stations.  By default\nnone filters are applied. Valid filters are:\n\n``language``\n  Filter stations by selected language.  For instance the ``de`` from ``:de-AU``\n  will be translated to `german` and used in the argument ``language=``.\n\n``countrycode``\n  Filter stations by selected country.  The 2-digit countrycode of the station\n  comes from the region the user selected.  For instance ``:de-AU`` will filter\n  out all stations not in ``AU``.\n\n.. note::\n\n   RadioBrowser has registered a lot of languages and countrycodes unknown to\n   :py:obj:`babel` and note that when searching for radio stations, users are\n   more likely to search by name than by region or language.\n\n'

def request(query, params):
    if False:
        return 10
    args = {'name': query, 'order': 'votes', 'offset': (params['pageno'] - 1) * number_of_results, 'limit': number_of_results, 'hidebroken': 'true', 'reverse': 'true'}
    if 'language' in station_filters:
        lang = traits.get_language(params['searxng_locale'])
        if lang:
            args['language'] = lang
    if 'countrycode' in station_filters:
        if len(params['searxng_locale'].split('-')) > 1:
            countrycode = params['searxng_locale'].split('-')[-1].upper()
            if countrycode in traits.custom['countrycodes']:
                args['countrycode'] = countrycode
    params['url'] = f'{base_url}/json/stations/search?{urlencode(args)}'
    return params

def response(resp):
    if False:
        return 10
    results = []
    json_resp = resp.json()
    for result in json_resp:
        url = result['homepage']
        if not url:
            url = result['url_resolved']
        content = []
        tags = ', '.join(result.get('tags', '').split(','))
        if tags:
            content.append(tags)
        for x in ['state', 'country']:
            v = result.get(x)
            if v:
                v = str(v).strip()
                content.append(v)
        metadata = []
        codec = result.get('codec')
        if codec and codec.lower() != 'unknown':
            metadata.append(f'{codec} ' + gettext('radio'))
        for (x, y) in [(gettext('bitrate'), 'bitrate'), (gettext('votes'), 'votes'), (gettext('clicks'), 'clickcount')]:
            v = result.get(y)
            if v:
                v = str(v).strip()
                metadata.append(f'{x} {v}')
        results.append({'url': url, 'title': result['name'], 'img_src': result.get('favicon', '').replace('http://', 'https://'), 'content': ' | '.join(content), 'metadata': ' | '.join(metadata), 'iframe_src': result['url_resolved'].replace('http://', 'https://')})
    return results

def fetch_traits(engine_traits: EngineTraits):
    if False:
        for i in range(10):
            print('nop')
    "Fetch languages and countrycodes from RadioBrowser\n\n    - ``traits.languages``: `list of languages API`_\n    - ``traits.custom['countrycodes']``: `list of countries API`_\n\n    .. _list of countries API: https://de1.api.radio-browser.info/#List_of_countries\n    .. _list of languages API: https://de1.api.radio-browser.info/#List_of_languages\n    "
    from babel.core import get_global
    babel_reg_list = get_global('territory_languages').keys()
    language_list = get(f'{base_url}/json/languages').json()
    country_list = get(f'{base_url}/json/countries').json()
    for lang in language_list:
        babel_lang = lang.get('iso_639')
        if not babel_lang:
            continue
        try:
            sxng_tag = language_tag(babel.Locale.parse(babel_lang, sep='-'))
        except babel.UnknownLocaleError:
            continue
        eng_tag = lang['name']
        conflict = engine_traits.languages.get(sxng_tag)
        if conflict:
            if conflict != eng_tag:
                print('CONFLICT: babel %s --> %s, %s' % (sxng_tag, conflict, eng_tag))
            continue
        engine_traits.languages[sxng_tag] = eng_tag
    countrycodes = set()
    for region in country_list:
        if region['iso_3166_1'] not in babel_reg_list:
            print(f"ERROR: region tag {region['iso_3166_1']} is unknown by babel")
            continue
        countrycodes.add(region['iso_3166_1'])
    countrycodes = list(countrycodes)
    countrycodes.sort()
    engine_traits.custom['countrycodes'] = countrycodes