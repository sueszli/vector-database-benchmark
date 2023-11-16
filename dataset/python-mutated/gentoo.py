"""
 Gentoo Wiki
"""
from urllib.parse import urlencode, urljoin
from lxml import html
from searx.utils import extract_text
about = {'website': 'https://wiki.gentoo.org/', 'wikidata_id': 'Q1050637', 'official_api_documentation': 'https://wiki.gentoo.org/api.php', 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['it', 'software wikis']
paging = True
base_url = 'https://wiki.gentoo.org'
xpath_results = '//ul[@class="mw-search-results"]/li'
xpath_link = './/div[@class="mw-search-result-heading"]/a'
xpath_content = './/div[@class="searchresult"]'

def locale_to_lang_code(locale):
    if False:
        while True:
            i = 10
    if locale.find('-') >= 0:
        locale = locale.split('-')[0]
    return locale
lang_urls = {'en': {'base': 'https://wiki.gentoo.org', 'search': '/index.php?title=Special:Search&offset={offset}&{query}'}, 'others': {'base': 'https://wiki.gentoo.org', 'search': '/index.php?title=Special:Search&offset={offset}&{query}                &profile=translation&languagefilter={language}'}}

def get_lang_urls(language):
    if False:
        i = 10
        return i + 15
    if language != 'en':
        return lang_urls['others']
    return lang_urls['en']
main_langs = {'ar': 'العربية', 'bg': 'Български', 'cs': 'Česky', 'da': 'Dansk', 'el': 'Ελληνικά', 'es': 'Español', 'he': 'עברית', 'hr': 'Hrvatski', 'hu': 'Magyar', 'it': 'Italiano', 'ko': '한국어', 'lt': 'Lietuviškai', 'nl': 'Nederlands', 'pl': 'Polski', 'pt': 'Português', 'ru': 'Русский', 'sl': 'Slovenský', 'th': 'ไทย', 'uk': 'Українська', 'zh': '简体中文'}

def request(query, params):
    if False:
        return 10
    language = locale_to_lang_code(params['language'])
    if language in main_langs:
        query += ' (' + main_langs[language] + ')'
    query = urlencode({'search': query})
    offset = (params['pageno'] - 1) * 20
    urls = get_lang_urls(language)
    search_url = urls['base'] + urls['search']
    params['url'] = search_url.format(query=query, offset=offset, language=language)
    return params

def response(resp):
    if False:
        while True:
            i = 10
    language = locale_to_lang_code(resp.search_params['language'])
    base_url = get_lang_urls(language)['base']
    results = []
    dom = html.fromstring(resp.text)
    for result in dom.xpath(xpath_results):
        link = result.xpath(xpath_link)[0]
        href = urljoin(base_url, link.attrib.get('href'))
        title = extract_text(link)
        content = extract_text(result.xpath(xpath_content))
        results.append({'url': href, 'title': title, 'content': content})
    return results