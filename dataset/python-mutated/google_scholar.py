"""This is the implementation of the Google Scholar engine.

Compared to other Google services the Scholar engine has a simple GET REST-API
and there does not exists `async` API.  Even though the API slightly vintage we
can make use of the :ref:`google API` to assemble the arguments of the GET
request.
"""
from typing import TYPE_CHECKING
from typing import Optional
from urllib.parse import urlencode
from datetime import datetime
from lxml import html
from searx.utils import eval_xpath, eval_xpath_getindex, eval_xpath_list, extract_text
from searx.exceptions import SearxEngineCaptchaException
from searx.engines.google import fetch_traits
from searx.engines.google import get_google_info, time_range_dict
from searx.enginelib.traits import EngineTraits
if TYPE_CHECKING:
    import logging
    logger: logging.Logger
traits: EngineTraits
about = {'website': 'https://scholar.google.com', 'wikidata_id': 'Q494817', 'official_api_documentation': 'https://developers.google.com/custom-search', 'use_official_api': False, 'require_api_key': False, 'results': 'HTML'}
categories = ['science', 'scientific publications']
paging = True
language_support = True
time_range_support = True
safesearch = False
send_accept_language_header = True

def time_range_args(params):
    if False:
        i = 10
        return i + 15
    "Returns a dictionary with a time range arguments based on\n    ``params['time_range']``.\n\n    Google Scholar supports a detailed search by year.  Searching by *last\n    month* or *last week* (as offered by SearXNG) is uncommon for scientific\n    publications and is not supported by Google Scholar.\n\n    To limit the result list when the users selects a range, all the SearXNG\n    ranges (*day*, *week*, *month*, *year*) are mapped to *year*.  If no range\n    is set an empty dictionary of arguments is returned.  Example;  when\n    user selects a time range (current year minus one in 2022):\n\n    .. code:: python\n\n        { 'as_ylo' : 2021 }\n\n    "
    ret_val = {}
    if params['time_range'] in time_range_dict:
        ret_val['as_ylo'] = datetime.now().year - 1
    return ret_val

def detect_google_captcha(dom):
    if False:
        i = 10
        return i + 15
    'In case of CAPTCHA Google Scholar open its own *not a Robot* dialog and is\n    not redirected to ``sorry.google.com``.\n    '
    if eval_xpath(dom, "//form[@id='gs_captcha_f']"):
        raise SearxEngineCaptchaException()

def request(query, params):
    if False:
        for i in range(10):
            print('nop')
    'Google-Scholar search request'
    google_info = get_google_info(params, traits)
    google_info['subdomain'] = google_info['subdomain'].replace('www.', 'scholar.')
    args = {'q': query, **google_info['params'], 'start': (params['pageno'] - 1) * 10, 'as_sdt': '2007', 'as_vis': '0'}
    args.update(time_range_args(params))
    params['url'] = 'https://' + google_info['subdomain'] + '/scholar?' + urlencode(args)
    params['cookies'] = google_info['cookies']
    params['headers'].update(google_info['headers'])
    return params

def parse_gs_a(text: Optional[str]):
    if False:
        while True:
            i = 10
    'Parse the text written in green.\n\n    Possible formats:\n    * "{authors} - {journal}, {year} - {publisher}"\n    * "{authors} - {year} - {publisher}"\n    * "{authors} - {publisher}"\n    '
    if text is None or text == '':
        return (None, None, None, None)
    s_text = text.split(' - ')
    authors = s_text[0].split(', ')
    publisher = s_text[-1]
    if len(s_text) != 3:
        return (authors, None, publisher, None)
    journal_year = s_text[1].split(', ')
    if len(journal_year) > 1:
        journal = ', '.join(journal_year[0:-1])
        if journal == '…':
            journal = None
    else:
        journal = None
    year = journal_year[-1]
    try:
        publishedDate = datetime.strptime(year.strip(), '%Y')
    except ValueError:
        publishedDate = None
    return (authors, journal, publisher, publishedDate)

def response(resp):
    if False:
        i = 10
        return i + 15
    'Parse response from Google Scholar'
    results = []
    dom = html.fromstring(resp.text)
    detect_google_captcha(dom)
    for result in eval_xpath_list(dom, '//div[@data-rp]'):
        title = extract_text(eval_xpath(result, './/h3[1]//a'))
        if not title:
            continue
        pub_type = extract_text(eval_xpath(result, './/span[@class="gs_ctg2"]'))
        if pub_type:
            pub_type = pub_type[1:-1].lower()
        url = eval_xpath_getindex(result, './/h3[1]//a/@href', 0)
        content = extract_text(eval_xpath(result, './/div[@class="gs_rs"]'))
        (authors, journal, publisher, publishedDate) = parse_gs_a(extract_text(eval_xpath(result, './/div[@class="gs_a"]')))
        if publisher in url:
            publisher = None
        comments = extract_text(eval_xpath(result, './/div[@class="gs_fl"]/a[starts-with(@href,"/scholar?cites=")]'))
        html_url = None
        pdf_url = None
        doc_url = eval_xpath_getindex(result, './/div[@class="gs_or_ggsm"]/a/@href', 0, default=None)
        doc_type = extract_text(eval_xpath(result, './/span[@class="gs_ctg2"]'))
        if doc_type == '[PDF]':
            pdf_url = doc_url
        else:
            html_url = doc_url
        results.append({'template': 'paper.html', 'type': pub_type, 'url': url, 'title': title, 'authors': authors, 'publisher': publisher, 'journal': journal, 'publishedDate': publishedDate, 'content': content, 'comments': comments, 'html_url': html_url, 'pdf_url': pdf_url})
    for suggestion in eval_xpath(dom, '//div[contains(@class, "gs_qsuggest_wrap")]//li//a'):
        results.append({'suggestion': extract_text(suggestion)})
    for correction in eval_xpath(dom, '//div[@class="gs_r gs_pda"]/a'):
        results.append({'correction': extract_text(correction)})
    return results