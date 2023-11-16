"""
Write a function that does the following:
Removes any duplicate query string parameters from the url
Removes any query string parameters specified within the 2nd argument (optional array)

An example:
www.saadbenn.com?a=1&b=2&a=2') // returns 'www.saadbenn.com?a=1&b=2'
"""
from collections import defaultdict
import urllib
import urllib.parse

def strip_url_params1(url, params_to_strip=None):
    if False:
        i = 10
        return i + 15
    if not params_to_strip:
        params_to_strip = []
    if url:
        result = ''
        tokens = url.split('?')
        domain = tokens[0]
        query_string = tokens[-1]
        result += domain
        if len(tokens) > 1:
            result += '?'
        if not query_string:
            return url
        else:
            key_value_string = []
            string = ''
            for char in query_string:
                if char.isdigit():
                    key_value_string.append(string + char)
                    string = ''
                else:
                    string += char
            dict = defaultdict(int)
            for i in key_value_string:
                _token = i.split('=')
                if _token[0]:
                    length = len(_token[0])
                    if length == 1:
                        if _token and (not _token[0] in dict):
                            if params_to_strip:
                                if _token[0] != params_to_strip[0]:
                                    dict[_token[0]] = _token[1]
                                    result = result + _token[0] + '=' + _token[1]
                            elif not _token[0] in dict:
                                dict[_token[0]] = _token[1]
                                result = result + _token[0] + '=' + _token[1]
                    else:
                        check = _token[0]
                        letter = check[1]
                        if _token and (not letter in dict):
                            if params_to_strip:
                                if letter != params_to_strip[0]:
                                    dict[letter] = _token[1]
                                    result = result + _token[0] + '=' + _token[1]
                            elif not letter in dict:
                                dict[letter] = _token[1]
                                result = result + _token[0] + '=' + _token[1]
    return result

def strip_url_params2(url, param_to_strip=[]):
    if False:
        while True:
            i = 10
    if '?' not in url:
        return url
    queries = url.split('?')[1].split('&')
    queries_obj = [query[0] for query in queries]
    for i in range(len(queries_obj) - 1, 0, -1):
        if queries_obj[i] in param_to_strip or queries_obj[i] in queries_obj[0:i]:
            queries.pop(i)
    return url.split('?')[0] + '?' + '&'.join(queries)

def strip_url_params3(url, strip=None):
    if False:
        print('Hello World!')
    if not strip:
        strip = []
    parse = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parse.query)
    query = {k: v[0] for (k, v) in query.items() if k not in strip}
    query = urllib.parse.urlencode(query)
    new = parse._replace(query=query)
    return new.geturl()