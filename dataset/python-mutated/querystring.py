import re
import urllib.parse

class UrlComponents:
    """
    `UrlComponents` are meant to be used internally for decoding-encoding, it has no external use
    """

    def _encode_url_component(self, url: str) -> str:
        if False:
            print('Hello World!')
        '\n        Function encodes querystring part of URL\n\n        Ex. q=dom & dogs -> q=dom+%26+dogs\n        '
        return urllib.parse.quote(url)

    def _decode_url_component(self, url: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Function decodes querystring part of URL\n\n        Ex. q=dom+%26+dogs -> q=dom & dogs\n        '
        return urllib.parse.unquote(url)

    def _is_encoded(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Function returns True if URL is already encoded\n        '
        if '?' in self.url:
            q_result = self._querystring_part()
            return self._decode_url_component(self.url[q_result.start() + 1:q_result.end()]) != self.url[q_result.start() + 1:q_result.end()]

    def _querystring_part(self, url_string: bool=False):
        if False:
            print('Hello World!')
        '\n        Function sliced url part and returns querystring part.\n\n        Use case: checking querystring part for encode, assiging decoded value\n        '
        pattern = re.compile('\\?[\\w\\D]+')
        data = pattern.search(self.url)
        return data if url_string is False else self.url[data.start() + 1:data.end()]

class QueryString(UrlComponents):
    """
    Note:
        `QueryString` class is meant to be for internal use inside of page. Hence, methods such as `get()` or `to_dict()` must be

        called from `page` object


    Constructor:
            `page` takes `Page` class an an argument and extracts URL automatically


    Methods:
            Public:
                `get()` method takes `key` an an argument and returns value according to key. (Ex: .../?name=Joe -> `get('name')` -> `Joe`)

                `to_dict` returns all the key-value pairs of querystring as a `dict`

                `path` returns url path (Ex: .../products?id=1 -> /products)

            Private(meant to be used only inside of page class):
                `post()` method takes key-value pair as an argument and returs proceeded querystring ready to be merged with URL

    """

    def __init__(self, page=None):
        if False:
            print('Hello World!')
        self.page = page
        self.url = None

    def get(self, key: str) -> str:
        if False:
            while True:
                i = 10
        self._data = self.to_dict
        return self._data[key]

    def post(self, kwargs: dict):
        if False:
            return 10
        return '?' + urllib.parse.urlencode(kwargs)

    @property
    def to_dict(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        self._data = urllib.parse.urlparse(self.url).query
        return dict(urllib.parse.parse_qsl(self._data))

    @property
    def path(self):
        if False:
            while True:
                i = 10
        self._updated_url = self.url.replace('#/', '') if '#' in self.url else self.url
        return urllib.parse.urlparse(self._updated_url).path

    def __call__(self):
        if False:
            return 10
        '\n        Call dunder method updates url after updating `Page`\n        '
        self.url = self.page.url + self.page.route
        if self._is_encoded() is True:
            self.url = self.page.url + urllib.parse.urlparse(self.url).path + '?' + self._decode_url_component(self._querystring_part(url_string=True))