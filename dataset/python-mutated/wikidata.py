"""This module implements the Wikidata engine.  Some implementations are shared
from :ref:`wikipedia engine`.

"""
from typing import TYPE_CHECKING
from hashlib import md5
from urllib.parse import urlencode, unquote
from json import loads
from dateutil.parser import isoparse
from babel.dates import format_datetime, format_date, format_time, get_datetime_format
from searx.data import WIKIDATA_UNITS
from searx.network import post, get
from searx.utils import searx_useragent, get_string_replaces_function
from searx.external_urls import get_external_url, get_earth_coordinates_url, area_to_osm_zoom
from searx.engines.wikipedia import fetch_wikimedia_traits, get_wiki_params
from searx.enginelib.traits import EngineTraits
if TYPE_CHECKING:
    import logging
    logger: logging.Logger
traits: EngineTraits
about = {'website': 'https://wikidata.org/', 'wikidata_id': 'Q2013', 'official_api_documentation': 'https://query.wikidata.org/', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
display_type = ['infobox']
'A list of display types composed from ``infobox`` and ``list``.  The latter\none will add a hit to the result list.  The first one will show a hit in the\ninfo box.  Both values can be set, or one of the two can be set.'
SPARQL_ENDPOINT_URL = 'https://query.wikidata.org/sparql'
SPARQL_EXPLAIN_URL = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql?explain'
WIKIDATA_PROPERTIES = {'P434': 'MusicBrainz', 'P435': 'MusicBrainz', 'P436': 'MusicBrainz', 'P966': 'MusicBrainz', 'P345': 'IMDb', 'P2397': 'YouTube', 'P1651': 'YouTube', 'P2002': 'Twitter', 'P2013': 'Facebook', 'P2003': 'Instagram'}
QUERY_TEMPLATE = '\nSELECT ?item ?itemLabel ?itemDescription ?lat ?long %SELECT%\nWHERE\n{\n  SERVICE wikibase:mwapi {\n        bd:serviceParam wikibase:endpoint "www.wikidata.org";\n        wikibase:api "EntitySearch";\n        wikibase:limit 1;\n        mwapi:search "%QUERY%";\n        mwapi:language "%LANGUAGE%".\n        ?item wikibase:apiOutputItem mwapi:item.\n  }\n  hint:Prior hint:runFirst "true".\n\n  %WHERE%\n\n  SERVICE wikibase:label {\n      bd:serviceParam wikibase:language "%LANGUAGE%,en".\n      ?item rdfs:label ?itemLabel .\n      ?item schema:description ?itemDescription .\n      %WIKIBASE_LABELS%\n  }\n\n}\nGROUP BY ?item ?itemLabel ?itemDescription ?lat ?long %GROUP_BY%\n'
QUERY_PROPERTY_NAMES = '\nSELECT ?item ?name\nWHERE {\n    {\n      SELECT ?item\n      WHERE { ?item wdt:P279* wd:Q12132 }\n    } UNION {\n      VALUES ?item { %ATTRIBUTES% }\n    }\n    OPTIONAL { ?item rdfs:label ?name. }\n}\n'
DUMMY_ENTITY_URLS = set(('http://www.wikidata.org/entity/' + wid for wid in ('Q4115189', 'Q13406268', 'Q15397819', 'Q17339402')))
sparql_string_escape = get_string_replaces_function({'\t': '\\\t', '\n': '\\\n', '\r': '\\\r', '\x08': '\\\x08', '\x0c': '\\\x0c', '"': '\\"', "'": "\\'", '\\': '\\\\'})
replace_http_by_https = get_string_replaces_function({'http:': 'https:'})

def get_headers():
    if False:
        while True:
            i = 10
    return {'Accept': 'application/sparql-results+json', 'User-Agent': searx_useragent()}

def get_label_for_entity(entity_id, language):
    if False:
        for i in range(10):
            print('nop')
    name = WIKIDATA_PROPERTIES.get(entity_id)
    if name is None:
        name = WIKIDATA_PROPERTIES.get((entity_id, language))
    if name is None:
        name = WIKIDATA_PROPERTIES.get((entity_id, language.split('-')[0]))
    if name is None:
        name = WIKIDATA_PROPERTIES.get((entity_id, 'en'))
    if name is None:
        name = entity_id
    return name

def send_wikidata_query(query, method='GET'):
    if False:
        i = 10
        return i + 15
    if method == 'GET':
        http_response = get(SPARQL_ENDPOINT_URL + '?' + urlencode({'query': query}), headers=get_headers())
    else:
        http_response = post(SPARQL_ENDPOINT_URL, data={'query': query}, headers=get_headers())
    if http_response.status_code != 200:
        logger.debug('SPARQL endpoint error %s', http_response.content.decode())
    logger.debug('request time %s', str(http_response.elapsed))
    http_response.raise_for_status()
    return loads(http_response.content.decode())

def request(query, params):
    if False:
        i = 10
        return i + 15
    (eng_tag, _wiki_netloc) = get_wiki_params(params['searxng_locale'], traits)
    (query, attributes) = get_query(query, eng_tag)
    logger.debug('request --> language %s // len(attributes): %s', eng_tag, len(attributes))
    params['method'] = 'POST'
    params['url'] = SPARQL_ENDPOINT_URL
    params['data'] = {'query': query}
    params['headers'] = get_headers()
    params['language'] = eng_tag
    params['attributes'] = attributes
    return params

def response(resp):
    if False:
        return 10
    results = []
    jsonresponse = loads(resp.content.decode())
    language = resp.search_params['language']
    attributes = resp.search_params['attributes']
    logger.debug('request --> language %s // len(attributes): %s', language, len(attributes))
    seen_entities = set()
    for result in jsonresponse.get('results', {}).get('bindings', []):
        attribute_result = {key: value['value'] for (key, value) in result.items()}
        entity_url = attribute_result['item']
        if entity_url not in seen_entities and entity_url not in DUMMY_ENTITY_URLS:
            seen_entities.add(entity_url)
            results += get_results(attribute_result, attributes, language)
        else:
            logger.debug('The SPARQL request returns duplicate entities: %s', str(attribute_result))
    return results
_IMG_SRC_DEFAULT_URL_PREFIX = 'https://commons.wikimedia.org/wiki/Special:FilePath/'
_IMG_SRC_NEW_URL_PREFIX = 'https://upload.wikimedia.org/wikipedia/commons/thumb/'

def get_thumbnail(img_src):
    if False:
        return 10
    'Get Thumbnail image from wikimedia commons\n\n    Images from commons.wikimedia.org are (HTTP) redirected to\n    upload.wikimedia.org.  The redirected URL can be calculated by this\n    function.\n\n    - https://stackoverflow.com/a/33691240\n\n    '
    logger.debug('get_thumbnail(): %s', img_src)
    if not img_src is None and _IMG_SRC_DEFAULT_URL_PREFIX in img_src.split()[0]:
        img_src_name = unquote(img_src.replace(_IMG_SRC_DEFAULT_URL_PREFIX, '').split('?', 1)[0].replace('%20', '_'))
        img_src_name_first = img_src_name
        img_src_name_second = img_src_name
        if '.svg' in img_src_name.split()[0]:
            img_src_name_second = img_src_name + '.png'
        img_src_size = img_src.replace(_IMG_SRC_DEFAULT_URL_PREFIX, '').split('?', 1)[1]
        img_src_size = img_src_size[img_src_size.index('=') + 1:img_src_size.index('&')]
        img_src_name_md5 = md5(img_src_name.encode('utf-8')).hexdigest()
        img_src = _IMG_SRC_NEW_URL_PREFIX + img_src_name_md5[0] + '/' + img_src_name_md5[0:2] + '/' + img_src_name_first + '/' + img_src_size + 'px-' + img_src_name_second
        logger.debug('get_thumbnail() redirected: %s', img_src)
    return img_src

def get_results(attribute_result, attributes, language):
    if False:
        while True:
            i = 10
    results = []
    infobox_title = attribute_result.get('itemLabel')
    infobox_id = attribute_result['item']
    infobox_id_lang = None
    infobox_urls = []
    infobox_attributes = []
    infobox_content = attribute_result.get('itemDescription', [])
    img_src = None
    img_src_priority = 0
    for attribute in attributes:
        value = attribute.get_str(attribute_result, language)
        if value is not None and value != '':
            attribute_type = type(attribute)
            if attribute_type in (WDURLAttribute, WDArticle):
                for url in value.split(', '):
                    infobox_urls.append({'title': attribute.get_label(language), 'url': url, **attribute.kwargs})
                    if 'list' in display_type and (attribute.kwargs.get('official') or attribute_type == WDArticle):
                        results.append({'title': infobox_title, 'url': url, 'content': infobox_content})
                    if attribute_type == WDArticle and (attribute.language == 'en' and infobox_id_lang is None or attribute.language != 'en'):
                        infobox_id_lang = attribute.language
                        infobox_id = url
            elif attribute_type == WDImageAttribute:
                if attribute.priority > img_src_priority:
                    img_src = get_thumbnail(value)
                    img_src_priority = attribute.priority
            elif attribute_type == WDGeoAttribute:
                area = attribute_result.get('P2046')
                osm_zoom = area_to_osm_zoom(area) if area else 19
                url = attribute.get_geo_url(attribute_result, osm_zoom=osm_zoom)
                if url:
                    infobox_urls.append({'title': attribute.get_label(language), 'url': url, 'entity': attribute.name})
            else:
                infobox_attributes.append({'label': attribute.get_label(language), 'value': value, 'entity': attribute.name})
    if infobox_id:
        infobox_id = replace_http_by_https(infobox_id)
    infobox_urls.append({'title': 'Wikidata', 'url': attribute_result['item']})
    if 'list' in display_type and img_src is None and (len(infobox_attributes) == 0) and (len(infobox_urls) == 1) and (len(infobox_content) == 0):
        results.append({'url': infobox_urls[0]['url'], 'title': infobox_title, 'content': infobox_content})
    elif 'infobox' in display_type:
        results.append({'infobox': infobox_title, 'id': infobox_id, 'content': infobox_content, 'img_src': img_src, 'urls': infobox_urls, 'attributes': infobox_attributes})
    return results

def get_query(query, language):
    if False:
        return 10
    attributes = get_attributes(language)
    select = [a.get_select() for a in attributes]
    where = list(filter(lambda s: len(s) > 0, [a.get_where() for a in attributes]))
    wikibase_label = list(filter(lambda s: len(s) > 0, [a.get_wikibase_label() for a in attributes]))
    group_by = list(filter(lambda s: len(s) > 0, [a.get_group_by() for a in attributes]))
    query = QUERY_TEMPLATE.replace('%QUERY%', sparql_string_escape(query)).replace('%SELECT%', ' '.join(select)).replace('%WHERE%', '\n  '.join(where)).replace('%WIKIBASE_LABELS%', '\n      '.join(wikibase_label)).replace('%GROUP_BY%', ' '.join(group_by)).replace('%LANGUAGE%', language)
    return (query, attributes)

def get_attributes(language):
    if False:
        print('Hello World!')
    attributes = []

    def add_value(name):
        if False:
            print('Hello World!')
        attributes.append(WDAttribute(name))

    def add_amount(name):
        if False:
            for i in range(10):
                print('nop')
        attributes.append(WDAmountAttribute(name))

    def add_label(name):
        if False:
            while True:
                i = 10
        attributes.append(WDLabelAttribute(name))

    def add_url(name, url_id=None, **kwargs):
        if False:
            while True:
                i = 10
        attributes.append(WDURLAttribute(name, url_id, kwargs))

    def add_image(name, url_id=None, priority=1):
        if False:
            for i in range(10):
                print('nop')
        attributes.append(WDImageAttribute(name, url_id, priority))

    def add_date(name):
        if False:
            print('Hello World!')
        attributes.append(WDDateAttribute(name))
    for p in ['P571', 'P576', 'P580', 'P582', 'P569', 'P570', 'P619', 'P620']:
        add_date(p)
    for p in ['P27', 'P495', 'P17', 'P159']:
        add_label(p)
    for p in ['P36', 'P35', 'P6', 'P122', 'P37']:
        add_label(p)
    add_value('P1082')
    add_amount('P2046')
    add_amount('P281')
    add_label('P38')
    add_amount('P2048')
    for p in ['P400', 'P50', 'P170', 'P57', 'P175', 'P178', 'P162', 'P176', 'P58', 'P272', 'P264', 'P123', 'P449', 'P750', 'P86']:
        add_label(p)
    add_date('P577')
    add_label('P136')
    add_label('P364')
    add_value('P212')
    add_value('P957')
    add_label('P275')
    add_label('P277')
    add_value('P348')
    add_label('P840')
    add_value('P1098')
    add_label('P282')
    add_label('P1018')
    add_value('P218')
    add_label('P169')
    add_label('P112')
    add_label('P1454')
    add_label('P137')
    add_label('P1029')
    add_label('P225')
    add_value('P274')
    add_label('P1346')
    add_value('P1120')
    add_value('P498')
    add_url('P856', official=True)
    attributes.append(WDArticle(language))
    if not language.startswith('en'):
        attributes.append(WDArticle('en'))
    add_url('P1324')
    add_url('P1581')
    add_url('P434', url_id='musicbrainz_artist')
    add_url('P435', url_id='musicbrainz_work')
    add_url('P436', url_id='musicbrainz_release_group')
    add_url('P966', url_id='musicbrainz_label')
    add_url('P345', url_id='imdb_id')
    add_url('P2397', url_id='youtube_channel')
    add_url('P1651', url_id='youtube_video')
    add_url('P2002', url_id='twitter_profile')
    add_url('P2013', url_id='facebook_profile')
    add_url('P2003', url_id='instagram_profile')
    attributes.append(WDGeoAttribute('P625'))
    add_image('P15', priority=1, url_id='wikimedia_image')
    add_image('P242', priority=2, url_id='wikimedia_image')
    add_image('P154', priority=3, url_id='wikimedia_image')
    add_image('P18', priority=4, url_id='wikimedia_image')
    add_image('P41', priority=5, url_id='wikimedia_image')
    add_image('P2716', priority=6, url_id='wikimedia_image')
    add_image('P2910', priority=7, url_id='wikimedia_image')
    return attributes

class WDAttribute:
    __slots__ = ('name',)

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def get_select(self):
        if False:
            while True:
                i = 10
        return '(group_concat(distinct ?{name};separator=", ") as ?{name}s)'.replace('{name}', self.name)

    def get_label(self, language):
        if False:
            while True:
                i = 10
        return get_label_for_entity(self.name, language)

    def get_where(self):
        if False:
            return 10
        return 'OPTIONAL { ?item wdt:{name} ?{name} . }'.replace('{name}', self.name)

    def get_wikibase_label(self):
        if False:
            for i in range(10):
                print('nop')
        return ''

    def get_group_by(self):
        if False:
            while True:
                i = 10
        return ''

    def get_str(self, result, language):
        if False:
            while True:
                i = 10
        return result.get(self.name + 's')

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<' + str(type(self).__name__) + ':' + self.name + '>'

class WDAmountAttribute(WDAttribute):

    def get_select(self):
        if False:
            while True:
                i = 10
        return '?{name} ?{name}Unit'.replace('{name}', self.name)

    def get_where(self):
        if False:
            i = 10
            return i + 15
        return '  OPTIONAL { ?item p:{name} ?{name}Node .\n    ?{name}Node rdf:type wikibase:BestRank ; ps:{name} ?{name} .\n    OPTIONAL { ?{name}Node psv:{name}/wikibase:quantityUnit ?{name}Unit. } }'.replace('{name}', self.name)

    def get_group_by(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_select()

    def get_str(self, result, language):
        if False:
            i = 10
            return i + 15
        value = result.get(self.name)
        unit = result.get(self.name + 'Unit')
        if unit is not None:
            unit = unit.replace('http://www.wikidata.org/entity/', '')
            return value + ' ' + get_label_for_entity(unit, language)
        return value

class WDArticle(WDAttribute):
    __slots__ = ('language', 'kwargs')

    def __init__(self, language, kwargs=None):
        if False:
            while True:
                i = 10
        super().__init__('wikipedia')
        self.language = language
        self.kwargs = kwargs or {}

    def get_label(self, language):
        if False:
            return 10
        return 'Wikipedia ({language})'.replace('{language}', self.language)

    def get_select(self):
        if False:
            while True:
                i = 10
        return '?article{language} ?articleName{language}'.replace('{language}', self.language)

    def get_where(self):
        if False:
            i = 10
            return i + 15
        return 'OPTIONAL { ?article{language} schema:about ?item ;\n             schema:inLanguage "{language}" ;\n             schema:isPartOf <https://{language}.wikipedia.org/> ;\n             schema:name ?articleName{language} . }'.replace('{language}', self.language)

    def get_group_by(self):
        if False:
            return 10
        return self.get_select()

    def get_str(self, result, language):
        if False:
            i = 10
            return i + 15
        key = 'article{language}'.replace('{language}', self.language)
        return result.get(key)

class WDLabelAttribute(WDAttribute):

    def get_select(self):
        if False:
            i = 10
            return i + 15
        return '(group_concat(distinct ?{name}Label;separator=", ") as ?{name}Labels)'.replace('{name}', self.name)

    def get_where(self):
        if False:
            for i in range(10):
                print('nop')
        return 'OPTIONAL { ?item wdt:{name} ?{name} . }'.replace('{name}', self.name)

    def get_wikibase_label(self):
        if False:
            while True:
                i = 10
        return '?{name} rdfs:label ?{name}Label .'.replace('{name}', self.name)

    def get_str(self, result, language):
        if False:
            i = 10
            return i + 15
        return result.get(self.name + 'Labels')

class WDURLAttribute(WDAttribute):
    HTTP_WIKIMEDIA_IMAGE = 'http://commons.wikimedia.org/wiki/Special:FilePath/'
    __slots__ = ('url_id', 'kwargs')

    def __init__(self, name, url_id=None, kwargs=None):
        if False:
            return 10
        super().__init__(name)
        self.url_id = url_id
        self.kwargs = kwargs

    def get_str(self, result, language):
        if False:
            print('Hello World!')
        value = result.get(self.name + 's')
        if self.url_id and value is not None and (value != ''):
            value = value.split(',')[0]
            url_id = self.url_id
            if value.startswith(WDURLAttribute.HTTP_WIKIMEDIA_IMAGE):
                value = value[len(WDURLAttribute.HTTP_WIKIMEDIA_IMAGE):]
                url_id = 'wikimedia_image'
            return get_external_url(url_id, value)
        return value

class WDGeoAttribute(WDAttribute):

    def get_label(self, language):
        if False:
            print('Hello World!')
        return 'OpenStreetMap'

    def get_select(self):
        if False:
            for i in range(10):
                print('nop')
        return '?{name}Lat ?{name}Long'.replace('{name}', self.name)

    def get_where(self):
        if False:
            while True:
                i = 10
        return 'OPTIONAL { ?item p:{name}/psv:{name} [\n    wikibase:geoLatitude ?{name}Lat ;\n    wikibase:geoLongitude ?{name}Long ] }'.replace('{name}', self.name)

    def get_group_by(self):
        if False:
            return 10
        return self.get_select()

    def get_str(self, result, language):
        if False:
            print('Hello World!')
        latitude = result.get(self.name + 'Lat')
        longitude = result.get(self.name + 'Long')
        if latitude and longitude:
            return latitude + ' ' + longitude
        return None

    def get_geo_url(self, result, osm_zoom=19):
        if False:
            i = 10
            return i + 15
        latitude = result.get(self.name + 'Lat')
        longitude = result.get(self.name + 'Long')
        if latitude and longitude:
            return get_earth_coordinates_url(latitude, longitude, osm_zoom)
        return None

class WDImageAttribute(WDURLAttribute):
    __slots__ = ('priority',)

    def __init__(self, name, url_id=None, priority=100):
        if False:
            print('Hello World!')
        super().__init__(name, url_id)
        self.priority = priority

class WDDateAttribute(WDAttribute):

    def get_select(self):
        if False:
            print('Hello World!')
        return '?{name} ?{name}timePrecision ?{name}timeZone ?{name}timeCalendar'.replace('{name}', self.name)

    def get_where(self):
        if False:
            for i in range(10):
                print('nop')
        return 'OPTIONAL { ?item p:{name}/psv:{name} [\n    wikibase:timeValue ?{name} ;\n    wikibase:timePrecision ?{name}timePrecision ;\n    wikibase:timeTimezone ?{name}timeZone ;\n    wikibase:timeCalendarModel ?{name}timeCalendar ] . }\n    hint:Prior hint:rangeSafe true;'.replace('{name}', self.name)

    def get_group_by(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_select()

    def format_8(self, value, locale):
        if False:
            while True:
                i = 10
        return value

    def format_9(self, value, locale):
        if False:
            for i in range(10):
                print('nop')
        year = int(value)
        if year < 1584:
            if year < 0:
                return str(year - 1)
            return str(year)
        timestamp = isoparse(value)
        return format_date(timestamp, format='yyyy', locale=locale)

    def format_10(self, value, locale):
        if False:
            i = 10
            return i + 15
        timestamp = isoparse(value)
        return format_date(timestamp, format='MMMM y', locale=locale)

    def format_11(self, value, locale):
        if False:
            return 10
        timestamp = isoparse(value)
        return format_date(timestamp, format='full', locale=locale)

    def format_13(self, value, locale):
        if False:
            return 10
        timestamp = isoparse(value)
        return get_datetime_format(format, locale=locale).replace("'", '').replace('{0}', format_time(timestamp, 'full', tzinfo=None, locale=locale)).replace('{1}', format_date(timestamp, 'short', locale=locale))

    def format_14(self, value, locale):
        if False:
            i = 10
            return i + 15
        return format_datetime(isoparse(value), format='full', locale=locale)
    DATE_FORMAT = {'0': ('format_8', 1000000000), '1': ('format_8', 100000000), '2': ('format_8', 10000000), '3': ('format_8', 1000000), '4': ('format_8', 100000), '5': ('format_8', 10000), '6': ('format_8', 1000), '7': ('format_8', 100), '8': ('format_8', 10), '9': ('format_9', 1), '10': ('format_10', 1), '11': ('format_11', 0), '12': ('format_13', 0), '13': ('format_13', 0), '14': ('format_14', 0)}

    def get_str(self, result, language):
        if False:
            for i in range(10):
                print('nop')
        value = result.get(self.name)
        if value == '' or value is None:
            return None
        precision = result.get(self.name + 'timePrecision')
        date_format = WDDateAttribute.DATE_FORMAT.get(precision)
        if date_format is not None:
            format_method = getattr(self, date_format[0])
            precision = date_format[1]
            try:
                if precision >= 1:
                    t = value.split('-')
                    if value.startswith('-'):
                        value = '-' + t[1]
                    else:
                        value = t[0]
                return format_method(value, language)
            except Exception:
                return value
        return value

def debug_explain_wikidata_query(query, method='GET'):
    if False:
        i = 10
        return i + 15
    if method == 'GET':
        http_response = get(SPARQL_EXPLAIN_URL + '&' + urlencode({'query': query}), headers=get_headers())
    else:
        http_response = post(SPARQL_EXPLAIN_URL, data={'query': query}, headers=get_headers())
    http_response.raise_for_status()
    return http_response.content

def init(engine_settings=None):
    if False:
        i = 10
        return i + 15
    WIKIDATA_PROPERTIES.update(WIKIDATA_UNITS)
    wikidata_property_names = []
    for attribute in get_attributes('en'):
        if type(attribute) in (WDAttribute, WDAmountAttribute, WDURLAttribute, WDDateAttribute, WDLabelAttribute):
            if attribute.name not in WIKIDATA_PROPERTIES:
                wikidata_property_names.append('wd:' + attribute.name)
    query = QUERY_PROPERTY_NAMES.replace('%ATTRIBUTES%', ' '.join(wikidata_property_names))
    jsonresponse = send_wikidata_query(query)
    for result in jsonresponse.get('results', {}).get('bindings', {}):
        name = result['name']['value']
        lang = result['name']['xml:lang']
        entity_id = result['item']['value'].replace('http://www.wikidata.org/entity/', '')
        WIKIDATA_PROPERTIES[entity_id, lang] = name.capitalize()

def fetch_traits(engine_traits: EngineTraits):
    if False:
        print('Hello World!')
    "Uses languages evaluated from :py:obj:`wikipedia.fetch_wikimedia_traits\n    <searx.engines.wikipedia.fetch_wikimedia_traits>` and removes\n\n    - ``traits.custom['wiki_netloc']``: wikidata does not have net-locations for\n      the languages and the list of all\n\n    - ``traits.custom['WIKIPEDIA_LANGUAGES']``: not used in the wikipedia engine\n\n    "
    fetch_wikimedia_traits(engine_traits)
    engine_traits.custom['wiki_netloc'] = {}
    engine_traits.custom['WIKIPEDIA_LANGUAGES'] = []