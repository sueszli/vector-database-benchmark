categories = ['general']

def request(query, params):
    if False:
        while True:
            i = 10
    "pre-request callback\n    params<dict>:\n      method  : POST/GET\n      headers : {}\n      data    : {} # if method == POST\n      url     : ''\n      category: 'search category'\n      pageno  : 1 # number of the requested page\n    "
    params['url'] = 'https://host/%s' % query
    return params

def response(resp):
    if False:
        for i in range(10):
            print('nop')
    'post-response callback\n    resp: requests response object\n    '
    return [{'url': '', 'title': '', 'content': ''}]