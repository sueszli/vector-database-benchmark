"""HTTP unit tests."""
from __future__ import print_function
import os
import pytest
import pytest_localserver
from workflow import web
TEST_DATA = [('baidu.html', {'Content-Type': 'text/html; charset=utf-8'}, 'utf-8'), ('us-ascii.xml', {'Content-Type': 'application/xml'}, 'us-ascii'), ('utf8.xml', {'Content-Type': 'application/xml'}, 'utf-8'), ('no-encoding.xml', {'Content-Type': 'application/xml'}, 'utf-8'), ('utf8.json', {'Content-Type': 'application/json'}, 'utf-8')]

def test_web_encoding(httpserver):
    if False:
        while True:
            i = 10
    'Test web encoding'
    test_data = []
    for (filename, headers, encoding) in TEST_DATA:
        p = os.path.join(os.path.dirname(__file__), 'data', filename)
        test_data.append((p, headers, encoding))
        p2 = '{0}.gz'.format(p)
        if os.path.exists(p2):
            h2 = headers.copy()
            h2['Content-Encoding'] = 'gzip'
            test_data.append((p2, h2, encoding))
    for (filepath, headers, encoding) in test_data:
        print('filepath={0!r}, headers={1!r}, encoding={2!r}'.format(filepath, headers, encoding))
        content = open(filepath).read()
        httpserver.serve_content(content, headers=headers)
        r = web.get(httpserver.url)
        r.raise_for_status()
        assert r.encoding == encoding
if __name__ == '__main__':
    pytest.main([__file__])