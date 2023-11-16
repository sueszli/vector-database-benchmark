"""
Jisho (the Japanese-English dictionary)
"""
from urllib.parse import urlencode, urljoin
about = {'website': 'https://jisho.org', 'wikidata_id': 'Q24568389', 'official_api_documentation': 'https://jisho.org/forum/54fefc1f6e73340b1f160000-is-there-any-kind-of-search-api', 'use_official_api': True, 'require_api_key': False, 'results': 'JSON', 'language': 'ja'}
categories = ['dictionaries']
paging = False
URL = 'https://jisho.org'
BASE_URL = 'https://jisho.org/word/'
SEARCH_URL = URL + '/api/v1/search/words?{query}'

def request(query, params):
    if False:
        print('Hello World!')
    query = urlencode({'keyword': query})
    params['url'] = SEARCH_URL.format(query=query)
    logger.debug(f"query_url --> {params['url']}")
    return params

def response(resp):
    if False:
        while True:
            i = 10
    results = []
    first_result = True
    search_results = resp.json()
    for page in search_results.get('data', []):
        parts_of_speech = page.get('senses') and page['senses'][0].get('parts_of_speech')
        if parts_of_speech and parts_of_speech[0] == 'Wikipedia definition':
            pass
        alt_forms = []
        for title_raw in page['japanese']:
            if 'word' not in title_raw:
                alt_forms.append(title_raw['reading'])
            else:
                title = title_raw['word']
                if 'reading' in title_raw:
                    title += ' (' + title_raw['reading'] + ')'
                alt_forms.append(title)
        result_url = urljoin(BASE_URL, page['slug'])
        definitions = get_definitions(page)
        content = ' '.join((f'{engdef}.' for (_, engdef, _) in definitions))
        results.append({'url': result_url, 'title': ', '.join(alt_forms), 'content': content[:300] + (content[300:] and '...')})
        if first_result:
            first_result = False
            results.append(get_infobox(alt_forms, result_url, definitions))
    return results

def get_definitions(page):
    if False:
        for i in range(10):
            print('nop')
    definitions = []
    for defn_raw in page['senses']:
        extra = []
        if defn_raw.get('tags'):
            if defn_raw.get('info'):
                extra.append(defn_raw['tags'][0] + ', ' + defn_raw['info'][0] + '. ')
            else:
                extra.append(', '.join(defn_raw['tags']) + '. ')
        elif defn_raw.get('info'):
            extra.append(', '.join(defn_raw['info']).capitalize() + '. ')
        if defn_raw.get('restrictions'):
            extra.append('Only applies to: ' + ', '.join(defn_raw['restrictions']) + '. ')
        definitions.append((', '.join(defn_raw['parts_of_speech']), '; '.join(defn_raw['english_definitions']), ''.join(extra)[:-1]))
    return definitions

def get_infobox(alt_forms, result_url, definitions):
    if False:
        for i in range(10):
            print('nop')
    infobox_content = []
    infobox_title = alt_forms[0]
    if len(alt_forms) > 1:
        infobox_content.append(f"<p><i>Other forms:</i> {', '.join(alt_forms[1:])}</p>")
    infobox_content.append('\n        <small><a href="https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project">JMdict</a> \n        and <a href="https://www.edrdg.org/enamdict/enamdict_doc.html">JMnedict</a> \n        by <a href="https://www.edrdg.org/edrdg/licence.html">EDRDG</a>, CC BY-SA 3.0.</small>\n        <ul>\n    ')
    for (pos, engdef, extra) in definitions:
        if pos == 'Wikipedia definition':
            infobox_content.append('</ul><small>Wikipedia, CC BY-SA 3.0.</small><ul>')
        pos = f'<i>{pos}</i>: ' if pos else ''
        extra = f' ({extra})' if extra else ''
        infobox_content.append(f'<li>{pos}{engdef}{extra}</li>')
    infobox_content.append('</ul>')
    return {'infobox': infobox_title, 'content': ''.join(infobox_content), 'urls': [{'title': 'Jisho.org', 'url': result_url}]}