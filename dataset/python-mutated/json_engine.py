from collections.abc import Iterable
from json import loads
from urllib.parse import urlencode
from searx.utils import to_string, html_to_text
search_url = None
url_query = None
content_query = None
title_query = None
content_html_to_text = False
title_html_to_text = False
paging = False
suggestion_query = ''
results_query = ''
cookies = {}
headers = {}
'Some engines might offer different result based on cookies or headers.\nPossible use-case: To set safesearch cookie or header to moderate.'
page_size = 1
first_page_num = 1

def iterate(iterable):
    if False:
        return 10
    if type(iterable) == dict:
        it = iterable.items()
    else:
        it = enumerate(iterable)
    for (index, value) in it:
        yield (str(index), value)

def is_iterable(obj):
    if False:
        for i in range(10):
            print('nop')
    if type(obj) == str:
        return False
    return isinstance(obj, Iterable)

def parse(query):
    if False:
        return 10
    q = []
    for part in query.split('/'):
        if part == '':
            continue
        else:
            q.append(part)
    return q

def do_query(data, q):
    if False:
        while True:
            i = 10
    ret = []
    if not q:
        return ret
    qkey = q[0]
    for (key, value) in iterate(data):
        if len(q) == 1:
            if key == qkey:
                ret.append(value)
            elif is_iterable(value):
                ret.extend(do_query(value, q))
        else:
            if not is_iterable(value):
                continue
            if key == qkey:
                ret.extend(do_query(value, q[1:]))
            else:
                ret.extend(do_query(value, q))
    return ret

def query(data, query_string):
    if False:
        while True:
            i = 10
    q = parse(query_string)
    return do_query(data, q)

def request(query, params):
    if False:
        i = 10
        return i + 15
    query = urlencode({'q': query})[2:]
    fp = {'query': query}
    if paging and search_url.find('{pageno}') >= 0:
        fp['pageno'] = (params['pageno'] - 1) * page_size + first_page_num
    params['cookies'].update(cookies)
    params['headers'].update(headers)
    params['url'] = search_url.format(**fp)
    params['query'] = query
    return params

def identity(arg):
    if False:
        i = 10
        return i + 15
    return arg

def response(resp):
    if False:
        print('Hello World!')
    results = []
    json = loads(resp.text)
    title_filter = html_to_text if title_html_to_text else identity
    content_filter = html_to_text if content_html_to_text else identity
    if results_query:
        rs = query(json, results_query)
        if not len(rs):
            return results
        for result in rs[0]:
            try:
                url = query(result, url_query)[0]
                title = query(result, title_query)[0]
            except:
                continue
            try:
                content = query(result, content_query)[0]
            except:
                content = ''
            results.append({'url': to_string(url), 'title': title_filter(to_string(title)), 'content': content_filter(to_string(content))})
    else:
        for (url, title, content) in zip(query(json, url_query), query(json, title_query), query(json, content_query)):
            results.append({'url': to_string(url), 'title': title_filter(to_string(title)), 'content': content_filter(to_string(content))})
    if not suggestion_query:
        return results
    for suggestion in query(json, suggestion_query):
        results.append({'suggestion': suggestion})
    return results