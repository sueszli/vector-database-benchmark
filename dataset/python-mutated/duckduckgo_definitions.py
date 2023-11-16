"""
DuckDuckGo Instant Answer API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `DDG-API <https://duckduckgo.com/api>`__ is no longer documented but from
reverse engineering we can see that some services (e.g. instant answers) still
in use from the DDG search engine.

As far we can say the *instant answers* API does not support languages, or at
least we could not find out how language support should work.  It seems that
most of the features are based on English terms.

"""
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse, urljoin
from lxml import html
from searx.data import WIKIDATA_UNITS
from searx.utils import extract_text, html_to_text, get_string_replaces_function
from searx.external_urls import get_external_url, get_earth_coordinates_url, area_to_osm_zoom
if TYPE_CHECKING:
    import logging
    logger: logging.Logger
about = {'website': 'https://duckduckgo.com/', 'wikidata_id': 'Q12805', 'official_api_documentation': 'https://duckduckgo.com/api', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON'}
send_accept_language_header = True
URL = 'https://api.duckduckgo.com/' + '?{query}&format=json&pretty=0&no_redirect=1&d=1'
WIKIDATA_PREFIX = ['http://www.wikidata.org/entity/', 'https://www.wikidata.org/entity/']
replace_http_by_https = get_string_replaces_function({'http:': 'https:'})

def is_broken_text(text):
    if False:
        i = 10
        return i + 15
    'duckduckgo may return something like ``<a href="xxxx">http://somewhere Related website<a/>``\n\n    The href URL is broken, the "Related website" may contains some HTML.\n\n    The best solution seems to ignore these results.\n    '
    return text.startswith('http') and ' ' in text

def result_to_text(text, htmlResult):
    if False:
        while True:
            i = 10
    result = None
    dom = html.fromstring(htmlResult)
    a = dom.xpath('//a')
    if len(a) >= 1:
        result = extract_text(a[0])
    else:
        result = text
    if not is_broken_text(result):
        return result
    return None

def request(query, params):
    if False:
        while True:
            i = 10
    params['url'] = URL.format(query=urlencode({'q': query}))
    return params

def response(resp):
    if False:
        print('Hello World!')
    results = []
    search_res = resp.json()
    content = ''
    heading = search_res.get('Heading', '')
    attributes = []
    urls = []
    infobox_id = None
    relatedTopics = []
    answer = search_res.get('Answer', '')
    if answer:
        logger.debug('AnswerType="%s" Answer="%s"', search_res.get('AnswerType'), answer)
        if search_res.get('AnswerType') not in ['calc', 'ip']:
            results.append({'answer': html_to_text(answer), 'url': search_res.get('AbstractURL', '')})
    if 'Definition' in search_res:
        content = content + search_res.get('Definition', '')
    if 'Abstract' in search_res:
        content = content + search_res.get('Abstract', '')
    image = search_res.get('Image')
    image = None if image == '' else image
    if image is not None and urlparse(image).netloc == '':
        image = urljoin('https://duckduckgo.com', image)
    for ddg_result in search_res.get('Results', []):
        firstURL = ddg_result.get('FirstURL')
        text = ddg_result.get('Text')
        if firstURL is not None and text is not None:
            urls.append({'title': text, 'url': firstURL})
            results.append({'title': heading, 'url': firstURL})
    for ddg_result in search_res.get('RelatedTopics', []):
        if 'FirstURL' in ddg_result:
            firstURL = ddg_result.get('FirstURL')
            text = ddg_result.get('Text')
            if not is_broken_text(text):
                suggestion = result_to_text(text, ddg_result.get('Result'))
                if suggestion != heading and suggestion is not None:
                    results.append({'suggestion': suggestion})
        elif 'Topics' in ddg_result:
            suggestions = []
            relatedTopics.append({'name': ddg_result.get('Name', ''), 'suggestions': suggestions})
            for topic_result in ddg_result.get('Topics', []):
                suggestion = result_to_text(topic_result.get('Text'), topic_result.get('Result'))
                if suggestion != heading and suggestion is not None:
                    suggestions.append(suggestion)
    abstractURL = search_res.get('AbstractURL', '')
    if abstractURL != '':
        infobox_id = abstractURL
        urls.append({'title': search_res.get('AbstractSource'), 'url': abstractURL, 'official': True})
        results.append({'url': abstractURL, 'title': heading})
    definitionURL = search_res.get('DefinitionURL', '')
    if definitionURL != '':
        infobox_id = definitionURL
        urls.append({'title': search_res.get('DefinitionSource'), 'url': definitionURL})
    if infobox_id:
        infobox_id = replace_http_by_https(infobox_id)
    if 'Infobox' in search_res:
        infobox = search_res.get('Infobox')
        if 'content' in infobox:
            osm_zoom = 17
            coordinates = None
            for info in infobox.get('content'):
                data_type = info.get('data_type')
                data_label = info.get('label')
                data_value = info.get('value')
                if data_value == '""':
                    continue
                external_url = get_external_url(data_type, data_value)
                if external_url is not None:
                    urls.append({'title': data_label, 'url': external_url})
                elif data_type in ['instance', 'wiki_maps_trigger', 'google_play_artist_id']:
                    pass
                elif data_type == 'string' and data_label == 'Website':
                    pass
                elif data_type == 'area':
                    attributes.append({'label': data_label, 'value': area_to_str(data_value), 'entity': 'P2046'})
                    osm_zoom = area_to_osm_zoom(data_value.get('amount'))
                elif data_type == 'coordinates':
                    if data_value.get('globe') == 'http://www.wikidata.org/entity/Q2':
                        coordinates = info
                    else:
                        attributes.append({'label': data_label, 'value': data_value, 'entity': 'P625'})
                elif data_type == 'string':
                    attributes.append({'label': data_label, 'value': data_value})
            if coordinates:
                data_label = coordinates.get('label')
                data_value = coordinates.get('value')
                latitude = data_value.get('latitude')
                longitude = data_value.get('longitude')
                url = get_earth_coordinates_url(latitude, longitude, osm_zoom)
                urls.append({'title': 'OpenStreetMap', 'url': url, 'entity': 'P625'})
    if len(heading) > 0:
        if image is None and len(attributes) == 0 and (len(urls) == 1) and (len(relatedTopics) == 0) and (len(content) == 0):
            results.append({'url': urls[0]['url'], 'title': heading, 'content': content})
        else:
            results.append({'infobox': heading, 'id': infobox_id, 'content': content, 'img_src': image, 'attributes': attributes, 'urls': urls, 'relatedTopics': relatedTopics})
    return results

def unit_to_str(unit):
    if False:
        for i in range(10):
            print('nop')
    for prefix in WIKIDATA_PREFIX:
        if unit.startswith(prefix):
            wikidata_entity = unit[len(prefix):]
            return WIKIDATA_UNITS.get(wikidata_entity, unit)
    return unit

def area_to_str(area):
    if False:
        i = 10
        return i + 15
    "parse ``{'unit': 'https://www.wikidata.org/entity/Q712226', 'amount': '+20.99'}``"
    unit = unit_to_str(area.get('unit'))
    if unit is not None:
        try:
            amount = float(area.get('amount'))
            return '{} {}'.format(amount, unit)
        except ValueError:
            pass
    return '{} {}'.format(area.get('amount', ''), area.get('unit', ''))