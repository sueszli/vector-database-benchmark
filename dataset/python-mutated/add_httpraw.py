import requests
from requests.sessions import Session
import json
from requests.structures import CaseInsensitiveDict

def extract_dict(text, sep, sep2='='):
    if False:
        while True:
            i = 10
    "Split the string into a dictionary according to the split method\n\n    :param text: Split text\n    :param sep: The first character of the split, usually'\n'\n    :param sep2: The second character of the split, the default is '='\n    :return: Return a dict type, the key is the 0th position of sep2,\n     and the value is the first position of sep2.\n     Only the text can be converted into a dictionary,\n     if the text is of other types, an error will occur\n    "
    _dict = CaseInsensitiveDict([l.split(sep2, 1) for l in text.split(sep)])
    return _dict

def httpraw(raw: str, ssl: bool=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send the original HTTP packet request, if you set the parameters such as headers in the parameters, the parameters\n    you set will be sent\n\n    :param raw: Original packet text\n    :param ssl: whether is HTTPS\n    :param kwargs: Support setting of parameters in requests\n    :return:requests.Response\n    '
    raw = raw.strip()
    raws = list(map(lambda x: x.strip(), raw.splitlines()))
    try:
        (method, path, protocol) = raws[0].split(' ')
    except Exception:
        raise Exception('Protocol format error')
    post = None
    _json = None
    if method.upper() == 'POST':
        index = 0
        for i in raws:
            index += 1
            if i.strip() == '':
                break
        if len(raws) == index:
            raise Exception
        tmp_headers = raws[1:index - 1]
        tmp_headers = extract_dict('\n'.join(tmp_headers), '\n', ': ')
        postData = '\n'.join(raws[index:])
        try:
            json.loads(postData)
            _json = postData
        except ValueError:
            post = postData
    else:
        tmp_headers = extract_dict('\n'.join(raws[1:]), '\n', ': ')
    netloc = 'http' if not ssl else 'https'
    host = tmp_headers.get('Host', None)
    if host is None:
        raise Exception('Host is None')
    del tmp_headers['Host']
    url = '{0}://{1}'.format(netloc, host + path)
    kwargs.setdefault('allow_redirects', True)
    kwargs.setdefault('data', post)
    kwargs.setdefault('headers', tmp_headers)
    kwargs.setdefault('json', _json)
    with Session() as session:
        return session.request(method=method, url=url, **kwargs)

def patch_addraw():
    if False:
        return 10
    requests.httpraw = httpraw