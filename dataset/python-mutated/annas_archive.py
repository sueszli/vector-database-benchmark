"""`Anna's Archive`_ is a free non-profit online shadow library metasearch
engine providing access to a variety of book resources (also via IPFS), created
by a team of anonymous archivists (AnnaArchivist_).

.. _Anna's Archive: https://annas-archive.org/
.. _AnnaArchivist: https://annas-software.org/AnnaArchivist/annas-archive

Configuration
=============

The engine has the following additional settings:

- :py:obj:`aa_content`
- :py:obj:`aa_ext`
- :py:obj:`aa_sort`

With this options a SearXNG maintainer is able to configure **additional**
engines for specific searches in Anna's Archive.  For example a engine to search
for *newest* articles and journals (PDF) / by shortcut ``!aaa <search-term>``.

.. code:: yaml

  - name: annas articles
    engine: annas_archive
    shortcut: aaa
    aa_content: 'journal_article'
    aa_ext: 'pdf'
    aa_sort: 'newest'

Implementations
===============

"""
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from lxml import html
from searx.utils import extract_text, eval_xpath, eval_xpath_list
from searx.enginelib.traits import EngineTraits
from searx.data import ENGINE_TRAITS
about: Dict[str, Any] = {'website': 'https://annas-archive.org/', 'wikidata_id': 'Q115288326', 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories: List[str] = ['files']
paging: bool = False
base_url: str = 'https://annas-archive.org'
aa_content: str = ''
"Anan's search form field **Content** / possible values::\n\n    journal_article, book_any, book_fiction, book_unknown, book_nonfiction,\n    book_comic, magazine, standards_document\n\nTo not filter use an empty string (default).\n"
aa_sort: str = ''
"Sort Anna's results, possible values::\n\n    newest, oldest, largest, smallest\n\nTo sort by *most relevant* use an empty string (default)."
aa_ext: str = ''
"Filter Anna's results by a file ending.  Common filters for example are\n``pdf`` and ``epub``.\n\n.. note::\n\n   Anna's Archive is a beta release: Filter results by file extension does not\n   really work on Anna's Archive.\n\n"

def init(engine_settings=None):
    if False:
        return 10
    "Check of engine's settings."
    traits = EngineTraits(**ENGINE_TRAITS['annas archive'])
    if aa_content and aa_content not in traits.custom['content']:
        raise ValueError(f'invalid setting content: {aa_content}')
    if aa_sort and aa_sort not in traits.custom['sort']:
        raise ValueError(f'invalid setting sort: {aa_sort}')
    if aa_ext and aa_ext not in traits.custom['ext']:
        raise ValueError(f'invalid setting ext: {aa_ext}')

def request(query, params: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        return 10
    q = quote(query)
    lang = traits.get_language(params['language'], traits.all_locale)
    params['url'] = base_url + f"/search?lang={lang or ''}&content={aa_content}&ext={aa_ext}&sort={aa_sort}&q={q}"
    return params

def response(resp) -> List[Dict[str, Optional[str]]]:
    if False:
        while True:
            i = 10
    results: List[Dict[str, Optional[str]]] = []
    dom = html.fromstring(resp.text)
    for item in eval_xpath_list(dom, '//main//div[contains(@class, "h-[125]")]/a'):
        results.append(_get_result(item))
    for item in eval_xpath_list(dom, '//main//div[contains(@class, "js-scroll-hidden")]'):
        item = html.fromstring(item.xpath('./comment()')[0].text)
        results.append(_get_result(item))
    return results

def _get_result(item):
    if False:
        for i in range(10):
            print('nop')
    return {'template': 'paper.html', 'url': base_url + item.xpath('./@href')[0], 'title': extract_text(eval_xpath(item, './/h3/text()[1]')), 'publisher': extract_text(eval_xpath(item, './/div[contains(@class, "text-sm")]')), 'authors': [extract_text(eval_xpath(item, './/div[contains(@class, "italic")]'))], 'content': extract_text(eval_xpath(item, './/div[contains(@class, "text-xs")]')), 'img_src': item.xpath('.//img/@src')[0]}

def fetch_traits(engine_traits: EngineTraits):
    if False:
        i = 10
        return i + 15
    "Fetch languages and other search arguments from Anna's search form."
    import babel
    from searx.network import get
    from searx.locales import language_tag
    engine_traits.all_locale = ''
    engine_traits.custom['content'] = []
    engine_traits.custom['ext'] = []
    engine_traits.custom['sort'] = []
    resp = get(base_url + '/search')
    if not resp.ok:
        raise RuntimeError("Response from Anna's search page is not OK.")
    dom = html.fromstring(resp.text)
    lang_map = {}
    for x in eval_xpath_list(dom, "//form//input[@name='lang']"):
        eng_lang = x.get('value')
        if eng_lang in ('', '_empty', 'nl-BE', 'und'):
            continue
        try:
            locale = babel.Locale.parse(lang_map.get(eng_lang, eng_lang), sep='-')
        except babel.UnknownLocaleError:
            continue
        sxng_lang = language_tag(locale)
        conflict = engine_traits.languages.get(sxng_lang)
        if conflict:
            if conflict != eng_lang:
                print('CONFLICT: babel %s --> %s, %s' % (sxng_lang, conflict, eng_lang))
            continue
        engine_traits.languages[sxng_lang] = eng_lang
    for x in eval_xpath_list(dom, "//form//input[@name='content']"):
        engine_traits.custom['content'].append(x.get('value'))
    for x in eval_xpath_list(dom, "//form//input[@name='ext']"):
        engine_traits.custom['ext'].append(x.get('value'))
    for x in eval_xpath_list(dom, "//form//select[@name='sort']//option"):
        engine_traits.custom['sort'].append(x.get('value'))