from requests.cookies import MockRequest

class MockResponse(object):

    def __init__(self, headers):
        if False:
            return 10
        self._headers = headers

    def info(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def getheaders(self, name):
        if False:
            i = 10
            return i + 15
        'make cookie python 2 version use this method to get cookie list'
        return self._headers.get_list(name)

    def get_all(self, name, default=None):
        if False:
            for i in range(10):
                print('nop')
        'make cookie python 3 version use this instead of getheaders'
        if default is None:
            default = []
        return self._headers.get_list(name) or default

def extract_cookies_to_jar(jar, request, response):
    if False:
        while True:
            i = 10
    req = MockRequest(request)
    res = MockResponse(response)
    jar.extract_cookies(res, req)