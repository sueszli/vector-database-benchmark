"""Brave supports the categories listed in :py:obj:`brave_category` (General,
news, videos, images).  The support of :py:obj:`paging` and :py:obj:`time range
<time_range_support>` is limited (see remarks).

Configured ``brave`` engines:

.. code:: yaml

  - name: brave
    engine: brave
    ...
    brave_category: search
    time_range_support: true
    paging: true

  - name: brave.images
    engine: brave
    ...
    brave_category: images

  - name: brave.videos
    engine: brave
    ...
    brave_category: videos

  - name: brave.news
    engine: brave
    ...
    brave_category: news


.. _brave regions:

Brave regions
=============

Brave uses two-digit tags for the regions like ``ca`` while SearXNG deals with
locales.  To get a mapping, all *officiat de-facto* languages of the Brave
region are mapped to regions in SearXNG (see :py:obj:`babel
<babel.languages.get_official_languages>`):

.. code:: python

    "regions": {
      ..
      "en-CA": "ca",
      "fr-CA": "ca",
      ..
     }


.. note::

   The language (aka region) support of Brave's index is limited to very basic
   languages.  The search results for languages like Chinese or Arabic are of
   low quality.


.. _brave languages:

Brave languages
===============

Brave's language support is limited to the UI (menus, area local notations,
etc).  Brave's index only seems to support a locale, but it does not seem to
support any languages in its index.  The choice of available languages is very
small (and its not clear to me where the difference in UI is when switching
from en-us to en-ca or en-gb).

In the :py:obj:`EngineTraits object <searx.enginelib.traits.EngineTraits>` the
UI languages are stored in a custom field named ``ui_lang``:

.. code:: python

    "custom": {
      "ui_lang": {
        "ca": "ca",
        "de-DE": "de-de",
        "en-CA": "en-ca",
        "en-GB": "en-gb",
        "en-US": "en-us",
        "es": "es",
        "fr-CA": "fr-ca",
        "fr-FR": "fr-fr",
        "ja-JP": "ja-jp",
        "pt-BR": "pt-br",
        "sq-AL": "sq-al"
      }
    },

Implementations
===============

"""
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse, parse_qs
from lxml import html
from searx import locales
from searx.utils import extract_text, eval_xpath_list, eval_xpath_getindex, js_variable_to_python
from searx.enginelib.traits import EngineTraits
if TYPE_CHECKING:
    import logging
    logger: logging.Logger
traits: EngineTraits
about = {'website': 'https://search.brave.com/', 'wikidata_id': 'Q22906900', 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
base_url = 'https://search.brave.com/'
categories = []
brave_category = 'search'
'Brave supports common web-search, video search, image and video search.\n\n- ``search``: Common WEB search\n- ``videos``: search for videos\n- ``images``: search for images\n- ``news``: search for news\n'
brave_spellcheck = False
'Brave supports some kind of spell checking.  When activated, Brave tries to\nfix typos, e.g. it searches for ``food`` when the user queries for ``fooh``.  In\nthe UI of Brave the user gets warned about this, since we can not warn the user\nin SearXNG, the spellchecking is disabled by default.\n'
send_accept_language_header = True
paging = False
'Brave only supports paging in :py:obj:`brave_category` ``search`` (UI\ncategory All).'
safesearch = True
safesearch_map = {2: 'strict', 1: 'moderate', 0: 'off'}
time_range_support = False
'Brave only supports time-range in :py:obj:`brave_category` ``search`` (UI\ncategory All).'
time_range_map = {'day': 'pd', 'week': 'pw', 'month': 'pm', 'year': 'py'}

def request(query, params):
    if False:
        for i in range(10):
            print('nop')
    params['headers']['Accept-Encoding'] = 'gzip, deflate'
    args = {'q': query}
    if brave_spellcheck:
        args['spellcheck'] = '1'
    if brave_category == 'search':
        if params.get('pageno', 1) - 1:
            args['offset'] = params.get('pageno', 1) - 1
        if time_range_map.get(params['time_range']):
            args['tf'] = time_range_map.get(params['time_range'])
    params['url'] = f'{base_url}{brave_category}?{urlencode(args)}'
    params['cookies']['safesearch'] = safesearch_map.get(params['safesearch'], 'off')
    params['cookies']['useLocation'] = '0'
    params['cookies']['summarizer'] = '0'
    engine_region = traits.get_region(params['searxng_locale'], 'all')
    params['cookies']['country'] = engine_region.split('-')[-1].lower()
    ui_lang = locales.get_engine_locale(params['searxng_locale'], traits.custom['ui_lang'], 'en-us')
    params['cookies']['ui_lang'] = ui_lang
    logger.debug('cookies %s', params['cookies'])

def response(resp):
    if False:
        return 10
    if brave_category == 'search':
        return _parse_search(resp)
    datastr = ''
    for line in resp.text.split('\n'):
        if 'const data = ' in line:
            datastr = line.replace('const data = ', '').strip()[:-1]
            break
    json_data = js_variable_to_python(datastr)
    json_resp = json_data[1]['data']['body']['response']
    if brave_category == 'news':
        return _parse_news(json_resp['news'])
    if brave_category == 'images':
        return _parse_images(json_resp)
    if brave_category == 'videos':
        return _parse_videos(json_resp)
    raise ValueError(f'Unsupported brave category: {brave_category}')

def _parse_search(resp):
    if False:
        while True:
            i = 10
    result_list = []
    dom = html.fromstring(resp.text)
    answer_tag = eval_xpath_getindex(dom, '//div[@class="answer"]', 0, default=None)
    if answer_tag:
        url = eval_xpath_getindex(dom, '//div[@id="featured_snippet"]/a[@class="result-header"]/@href', 0, default=None)
        result_list.append({'answer': extract_text(answer_tag), 'url': url})
    xpath_results = '//div[contains(@class, "snippet ")]'
    for result in eval_xpath_list(dom, xpath_results):
        url = eval_xpath_getindex(result, './/a[contains(@class, "h")]/@href', 0, default=None)
        title_tag = eval_xpath_getindex(result, './/div[contains(@class, "url")]', 0, default=None)
        if url is None or title_tag is None or (not urlparse(url).netloc):
            continue
        content_tag = eval_xpath_getindex(result, './/div[@class="snippet-description"]', 0, default='')
        img_src = eval_xpath_getindex(result, './/img[contains(@class, "thumb")]/@src', 0, default='')
        item = {'url': url, 'title': extract_text(title_tag), 'content': extract_text(content_tag), 'img_src': img_src}
        video_tag = eval_xpath_getindex(result, './/div[contains(@class, "video-snippet") and @data-macro="video"]', 0, default=None)
        if video_tag is not None:
            iframe_src = _get_iframe_src(url)
            if iframe_src:
                item['iframe_src'] = iframe_src
                item['template'] = 'videos.html'
                item['thumbnail'] = eval_xpath_getindex(video_tag, './/img/@src', 0, default='')
            else:
                item['img_src'] = eval_xpath_getindex(video_tag, './/img/@src', 0, default='')
        result_list.append(item)
    return result_list

def _get_iframe_src(url):
    if False:
        print('Hello World!')
    parsed_url = urlparse(url)
    if parsed_url.path == '/watch' and parsed_url.query:
        video_id = parse_qs(parsed_url.query).get('v', [])
        if video_id:
            return 'https://www.youtube-nocookie.com/embed/' + video_id[0]
    return None

def _parse_news(json_resp):
    if False:
        for i in range(10):
            print('nop')
    result_list = []
    for result in json_resp['results']:
        item = {'url': result['url'], 'title': result['title'], 'content': result['description']}
        if result['thumbnail'] is not None:
            item['img_src'] = result['thumbnail']['src']
        result_list.append(item)
    return result_list

def _parse_images(json_resp):
    if False:
        i = 10
        return i + 15
    result_list = []
    for result in json_resp['results']:
        item = {'url': result['url'], 'title': result['title'], 'content': result['description'], 'template': 'images.html', 'img_format': result['properties']['format'], 'source': result['source'], 'img_src': result['properties']['url']}
        result_list.append(item)
    return result_list

def _parse_videos(json_resp):
    if False:
        i = 10
        return i + 15
    result_list = []
    for result in json_resp['results']:
        url = result['url']
        item = {'url': url, 'title': result['title'], 'content': result['description'], 'template': 'videos.html', 'length': result['video']['duration'], 'duration': result['video']['duration']}
        if result['thumbnail'] is not None:
            item['thumbnail'] = result['thumbnail']['src']
        iframe_src = _get_iframe_src(url)
        if iframe_src:
            item['iframe_src'] = iframe_src
        result_list.append(item)
    return result_list

def fetch_traits(engine_traits: EngineTraits):
    if False:
        return 10
    'Fetch :ref:`languages <brave languages>` and :ref:`regions <brave\n    regions>` from Brave.'
    import babel.languages
    from searx.locales import region_tag, language_tag
    from searx.network import get
    engine_traits.custom['ui_lang'] = {}
    headers = {'Accept-Encoding': 'gzip, deflate'}
    lang_map = {'no': 'nb'}
    resp = get('https://search.brave.com/settings', headers=headers)
    if not resp.ok:
        print('ERROR: response from Brave is not OK.')
    dom = html.fromstring(resp.text)
    for option in dom.xpath('//div[@id="language-select"]//option'):
        ui_lang = option.get('value')
        try:
            if '-' in ui_lang:
                sxng_tag = region_tag(babel.Locale.parse(ui_lang, sep='-'))
            else:
                sxng_tag = language_tag(babel.Locale.parse(ui_lang))
        except babel.UnknownLocaleError:
            print("ERROR: can't determine babel locale of Brave's (UI) language %s" % ui_lang)
            continue
        conflict = engine_traits.custom['ui_lang'].get(sxng_tag)
        if conflict:
            if conflict != ui_lang:
                print('CONFLICT: babel %s --> %s, %s' % (sxng_tag, conflict, ui_lang))
            continue
        engine_traits.custom['ui_lang'][sxng_tag] = ui_lang
    resp = get('https://cdn.search.brave.com/serp/v2/_app/immutable/chunks/parameters.734c106a.js', headers=headers)
    if not resp.ok:
        print('ERROR: response from Brave is not OK.')
    country_js = resp.text[resp.text.index('options:{all') + len('options:'):]
    country_js = country_js[:country_js.index('},k={default')]
    country_tags = js_variable_to_python(country_js)
    for (k, v) in country_tags.items():
        if k == 'all':
            engine_traits.all_locale = 'all'
            continue
        country_tag = v['value']
        for lang_tag in babel.languages.get_official_languages(country_tag, de_facto=True):
            lang_tag = lang_map.get(lang_tag, lang_tag)
            sxng_tag = region_tag(babel.Locale.parse('%s_%s' % (lang_tag, country_tag.upper())))
            conflict = engine_traits.regions.get(sxng_tag)
            if conflict:
                if conflict != country_tag:
                    print('CONFLICT: babel %s --> %s, %s' % (sxng_tag, conflict, country_tag))
                    continue
            engine_traits.regions[sxng_tag] = country_tag