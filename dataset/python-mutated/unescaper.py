"""URL unescaper functions."""
from xml.sax import saxutils
__all__ = ['unescape_all']
_bytes_entities = {b'&amp;': b'&', b'&lt;': b'<', b'&gt;': b'>', b'&amp;&amp;': b'&', b'&&': b'&', b'%2F': b'/'}
_bytes_keys = [b'&amp;&amp;', b'&&', b'&amp;', b'&lt;', b'&gt;', b'%2F']
_str_entities = {'&amp;&amp;': '&', '&&': '&', '%2F': '/'}
_str_keys = ['&amp;&amp;', '&&', '&amp;', '&lt;', '&gt;', '%2F']

def unescape_all(url):
    if False:
        return 10
    "Recursively unescape a given URL.\n\n    .. note:: '&amp;&amp;' becomes a single '&'.\n\n    Parameters\n    ----------\n    url : str or bytes\n        URL to unescape.\n\n    Returns\n    -------\n    clean_url : str or bytes\n        Unescaped URL.\n\n    "
    if isinstance(url, bytes):
        func2use = _unescape_bytes
        keys2use = _bytes_keys
    else:
        func2use = _unescape_str
        keys2use = _str_keys
    clean_url = func2use(url)
    not_done = [clean_url.count(key) > 0 for key in keys2use]
    if True in not_done:
        return unescape_all(clean_url)
    else:
        return clean_url

def _unescape_str(url):
    if False:
        for i in range(10):
            print('nop')
    return saxutils.unescape(url, _str_entities)

def _unescape_bytes(url):
    if False:
        print('Hello World!')
    clean_url = url
    for key in _bytes_keys:
        clean_url = clean_url.replace(key, _bytes_entities[key])
    return clean_url