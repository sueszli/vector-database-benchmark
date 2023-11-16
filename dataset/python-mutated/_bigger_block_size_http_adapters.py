import sys
from requests.adapters import HTTPAdapter

class BiggerBlockSizeHTTPAdapter(HTTPAdapter):

    def get_connection(self, url, proxies=None):
        if False:
            print('Hello World!')
        'Returns a urllib3 connection for the given URL. This should not be\n        called from user code, and is only exposed for use when subclassing the\n        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.\n\n        :param str url: The URL to connect to.\n        :param dict proxies: (optional) A Requests-style dictionary of proxies used on this request.\n        :rtype: urllib3.ConnectionPool\n        :returns: The urllib3 ConnectionPool for the given URL.\n        '
        conn = super(BiggerBlockSizeHTTPAdapter, self).get_connection(url, proxies)
        system_version = tuple(sys.version_info)[:3]
        if system_version[:2] >= (3, 7):
            if not conn.conn_kw:
                conn.conn_kw = {}
            conn.conn_kw['blocksize'] = 32768
        return conn