"""
DuckDuckGo Lite
~~~~~~~~~~~~~~~
"""
from typing import TYPE_CHECKING
import re
from urllib.parse import urlencode
import json
import babel
import lxml.html
from searx import locales, redislib, external_bang
from searx.utils import eval_xpath, eval_xpath_getindex, extract_text
from searx.network import get
from searx import redisdb
from searx.enginelib.traits import EngineTraits
if TYPE_CHECKING:
    import logging
    logger: logging.Logger
traits: EngineTraits
about = {'website': 'https://lite.duckduckgo.com/lite/', 'wikidata_id': 'Q12805', 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
send_accept_language_header = True
"DuckDuckGo-Lite tries to guess user's prefered language from the HTTP\n``Accept-Language``.  Optional the user can select a region filter (but not a\nlanguage).\n"
categories = ['general', 'web']
paging = True
time_range_support = True
safesearch = True
url = 'https://lite.duckduckgo.com/lite/'
time_range_dict = {'day': 'd', 'week': 'w', 'month': 'm', 'year': 'y'}
form_data = {'v': 'l', 'api': 'd.js', 'o': 'json'}

def cache_vqd(query, value):
    if False:
        return 10
    'Caches a ``vqd`` value from a query.'
    c = redisdb.client()
    if c:
        logger.debug('cache vqd value: %s', value)
        key = 'SearXNG_ddg_vqd' + redislib.secret_hash(query)
        c.set(key, value, ex=600)

def get_vqd(query):
    if False:
        return 10
    "Returns the ``vqd`` that fits to the *query*.  If there is no ``vqd`` cached\n    (:py:obj:`cache_vqd`) the query is sent to DDG to get a vqd value from the\n    response.\n\n    .. hint::\n\n       If an empty string is returned there are no results for the ``query`` and\n       therefore no ``vqd`` value.\n\n    DDG's bot detection is sensitive to the ``vqd`` value.  For some search terms\n    (such as extremely long search terms that are often sent by bots), no ``vqd``\n    value can be determined.\n\n    If SearXNG cannot determine a ``vqd`` value, then no request should go out\n    to DDG:\n\n        A request with a wrong ``vqd`` value leads to DDG temporarily putting\n        SearXNG's IP on a block list.\n\n        Requests from IPs in this block list run into timeouts.\n\n    Not sure, but it seems the block list is a sliding window: to get my IP rid\n    from the bot list I had to cool down my IP for 1h (send no requests from\n    that IP to DDG).\n\n    TL;DR; the ``vqd`` value is needed to pass DDG's bot protection and is used\n    by all request to DDG:\n\n    - DuckDuckGo Lite: ``https://lite.duckduckgo.com/lite`` (POST form data)\n    - DuckDuckGo Web: ``https://links.duckduckgo.com/d.js?q=...&vqd=...``\n    - DuckDuckGo Images: ``https://duckduckgo.com/i.js??q=...&vqd=...``\n    - DuckDuckGo Videos: ``https://duckduckgo.com/v.js??q=...&vqd=...``\n    - DuckDuckGo News: ``https://duckduckgo.com/news.js??q=...&vqd=...``\n\n    "
    value = ''
    c = redisdb.client()
    if c:
        key = 'SearXNG_ddg_vqd' + redislib.secret_hash(query)
        value = c.get(key)
        if value or value == b'':
            value = value.decode('utf-8')
            logger.debug('re-use cached vqd value: %s', value)
            return value
    query_url = 'https://lite.duckduckgo.com/lite/?{args}'.format(args=urlencode({'q': query}))
    res = get(query_url)
    doc = lxml.html.fromstring(res.text)
    value = doc.xpath("//input[@name='vqd']/@value")
    if value:
        value = value[0]
    else:
        value = ''
    logger.debug("new vqd value: '%s'", value)
    cache_vqd(query, value)
    return value

def get_ddg_lang(eng_traits: EngineTraits, sxng_locale, default='en_US'):
    if False:
        while True:
            i = 10
    "Get DuckDuckGo's language identifier from SearXNG's locale.\n\n    DuckDuckGo defines its languages by region codes (see\n    :py:obj:`fetch_traits`).\n\n    To get region and language of a DDG service use:\n\n    .. code: python\n\n       eng_region = traits.get_region(params['searxng_locale'], traits.all_locale)\n       eng_lang = get_ddg_lang(traits, params['searxng_locale'])\n\n    It might confuse, but the ``l`` value of the cookie is what SearXNG calls\n    the *region*:\n\n    .. code:: python\n\n        # !ddi paris :es-AR --> {'ad': 'es_AR', 'ah': 'ar-es', 'l': 'ar-es'}\n        params['cookies']['ad'] = eng_lang\n        params['cookies']['ah'] = eng_region\n        params['cookies']['l'] = eng_region\n\n    .. hint::\n\n       `DDG-lite <https://lite.duckduckgo.com/lite>`__ does not offer a language\n       selection to the user, only a region can be selected by the user\n       (``eng_region`` from the example above).  DDG-lite stores the selected\n       region in a cookie::\n\n         params['cookies']['kl'] = eng_region  # 'ar-es'\n\n    "
    return eng_traits.custom['lang_region'].get(sxng_locale, eng_traits.get_language(sxng_locale, default))
ddg_reg_map = {'tw-tzh': 'zh_TW', 'hk-tzh': 'zh_HK', 'ct-ca': 'skip', 'es-ca': 'ca_ES', 'id-en': 'id_ID', 'no-no': 'nb_NO', 'jp-jp': 'ja_JP', 'kr-kr': 'ko_KR', 'xa-ar': 'ar_SA', 'sl-sl': 'sl_SI', 'th-en': 'th_TH', 'vn-en': 'vi_VN'}
ddg_lang_map = {'ar_DZ': 'lang_region', 'ar_JO': 'lang_region', 'ar_SA': 'lang_region', 'bn_IN': 'lang_region', 'de_CH': 'lang_region', 'en_AU': 'lang_region', 'en_CA': 'lang_region', 'en_GB': 'lang_region', 'eo_XX': 'eo', 'es_AR': 'lang_region', 'es_CL': 'lang_region', 'es_CO': 'lang_region', 'es_CR': 'lang_region', 'es_EC': 'lang_region', 'es_MX': 'lang_region', 'es_PE': 'lang_region', 'es_UY': 'lang_region', 'es_VE': 'lang_region', 'fr_CA': 'lang_region', 'fr_CH': 'lang_region', 'fr_BE': 'lang_region', 'nl_BE': 'lang_region', 'pt_BR': 'lang_region', 'od_IN': 'skip', 'io_XX': 'skip', 'tokipona_XX': 'skip'}

def request(query, params):
    if False:
        print('Hello World!')
    vqd = get_vqd(query)
    if not vqd:
        params['url'] = None
        return params
    query_parts = []
    for val in re.split('(\\s+)', query):
        if not val.strip():
            continue
        if val.startswith('!') and external_bang.get_node(external_bang.EXTERNAL_BANGS, val[1:]):
            val = f"'{val}'"
        query_parts.append(val)
    query = ' '.join(query_parts)
    eng_region = traits.get_region(params['searxng_locale'], traits.all_locale)
    params['url'] = url
    params['method'] = 'POST'
    params['data']['q'] = query
    params['headers']['Content-Type'] = 'application/x-www-form-urlencoded'
    params['data']['vqd'] = vqd
    if params['pageno'] == 2:
        offset = (params['pageno'] - 1) * 30
        params['data']['s'] = offset
        params['data']['dc'] = offset + 1
    elif params['pageno'] > 2:
        offset = 30 + (params['pageno'] - 2) * 50
        params['data']['s'] = offset
        params['data']['dc'] = offset + 1
    if params['pageno'] > 1:
        params['data']['o'] = form_data.get('o', 'json')
        params['data']['api'] = form_data.get('api', 'd.js')
        params['data']['nextParams'] = form_data.get('nextParams', '')
        params['data']['v'] = form_data.get('v', 'l')
        params['headers']['Referer'] = 'https://lite.duckduckgo.com/'
    params['data']['kl'] = eng_region
    params['cookies']['kl'] = eng_region
    params['data']['df'] = ''
    if params['time_range'] in time_range_dict:
        params['data']['df'] = time_range_dict[params['time_range']]
        params['cookies']['df'] = time_range_dict[params['time_range']]
    logger.debug('param data: %s', params['data'])
    logger.debug('param cookies: %s', params['cookies'])
    return params

def response(resp):
    if False:
        while True:
            i = 10
    if resp.status_code == 303:
        return []
    results = []
    doc = lxml.html.fromstring(resp.text)
    result_table = eval_xpath(doc, '//html/body/form/div[@class="filters"]/table')
    if len(result_table) == 2:
        result_table = result_table[1]
    elif not len(result_table) >= 3:
        return []
    else:
        result_table = result_table[2]
        form = eval_xpath(doc, '//html/body/form/div[@class="filters"]/table//input/..')
        if len(form):
            form = form[0]
            form_data['v'] = eval_xpath(form, '//input[@name="v"]/@value')[0]
            form_data['api'] = eval_xpath(form, '//input[@name="api"]/@value')[0]
            form_data['o'] = eval_xpath(form, '//input[@name="o"]/@value')[0]
            logger.debug('form_data: %s', form_data)
            value = eval_xpath(form, '//input[@name="vqd"]/@value')[0]
            query = resp.search_params['data']['q']
            cache_vqd(query, value)
    tr_rows = eval_xpath(result_table, './/tr')
    tr_rows = tr_rows[:-1]
    len_tr_rows = len(tr_rows)
    offset = 0
    while len_tr_rows >= offset + 4:
        tr_title = tr_rows[offset]
        tr_content = tr_rows[offset + 1]
        offset += 4
        if tr_content.get('class') == 'result-sponsored':
            continue
        a_tag = eval_xpath_getindex(tr_title, './/td//a[@class="result-link"]', 0, None)
        if a_tag is None:
            continue
        td_content = eval_xpath_getindex(tr_content, './/td[@class="result-snippet"]', 0, None)
        if td_content is None:
            continue
        results.append({'title': a_tag.text_content(), 'content': extract_text(td_content), 'url': a_tag.get('href')})
    return results

def fetch_traits(engine_traits: EngineTraits):
    if False:
        i = 10
        return i + 15
    'Fetch languages & regions from DuckDuckGo.\n\n    SearXNG\'s ``all`` locale maps DuckDuckGo\'s "Alle regions" (``wt-wt``).\n    DuckDuckGo\'s language "Browsers prefered language" (``wt_WT``) makes no\n    sense in a SearXNG request since SearXNG\'s ``all`` will not add a\n    ``Accept-Language`` HTTP header.  The value in ``engine_traits.all_locale``\n    is ``wt-wt`` (the region).\n\n    Beside regions DuckDuckGo also defines its languages by region codes.  By\n    example these are the english languages in DuckDuckGo:\n\n    - en_US\n    - en_AU\n    - en_CA\n    - en_GB\n\n    The function :py:obj:`get_ddg_lang` evaluates DuckDuckGo\'s language from\n    SearXNG\'s locale.\n\n    '
    engine_traits.all_locale = 'wt-wt'
    resp = get('https://duckduckgo.com/util/u661.js')
    if not resp.ok:
        print('ERROR: response from DuckDuckGo is not OK.')
    pos = resp.text.find('regions:{') + 8
    js_code = resp.text[pos:]
    pos = js_code.find('}') + 1
    regions = json.loads(js_code[:pos])
    for (eng_tag, name) in regions.items():
        if eng_tag == 'wt-wt':
            engine_traits.all_locale = 'wt-wt'
            continue
        region = ddg_reg_map.get(eng_tag)
        if region == 'skip':
            continue
        if not region:
            (eng_territory, eng_lang) = eng_tag.split('-')
            region = eng_lang + '_' + eng_territory.upper()
        try:
            sxng_tag = locales.region_tag(babel.Locale.parse(region))
        except babel.UnknownLocaleError:
            print('ERROR: %s (%s) -> %s is unknown by babel' % (name, eng_tag, region))
            continue
        conflict = engine_traits.regions.get(sxng_tag)
        if conflict:
            if conflict != eng_tag:
                print('CONFLICT: babel %s --> %s, %s' % (sxng_tag, conflict, eng_tag))
            continue
        engine_traits.regions[sxng_tag] = eng_tag
    engine_traits.custom['lang_region'] = {}
    pos = resp.text.find('languages:{') + 10
    js_code = resp.text[pos:]
    pos = js_code.find('}') + 1
    js_code = '{"' + js_code[1:pos].replace(':', '":').replace(',', ',"')
    languages = json.loads(js_code)
    for (eng_lang, name) in languages.items():
        if eng_lang == 'wt_WT':
            continue
        babel_tag = ddg_lang_map.get(eng_lang, eng_lang)
        if babel_tag == 'skip':
            continue
        try:
            if babel_tag == 'lang_region':
                sxng_tag = locales.region_tag(babel.Locale.parse(eng_lang))
                engine_traits.custom['lang_region'][sxng_tag] = eng_lang
                continue
            sxng_tag = locales.language_tag(babel.Locale.parse(babel_tag))
        except babel.UnknownLocaleError:
            print('ERROR: language %s (%s) is unknown by babel' % (name, eng_lang))
            continue
        conflict = engine_traits.languages.get(sxng_tag)
        if conflict:
            if conflict != eng_lang:
                print('CONFLICT: babel %s --> %s, %s' % (sxng_tag, conflict, eng_lang))
            continue
        engine_traits.languages[sxng_tag] = eng_lang