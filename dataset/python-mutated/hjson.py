"""
hjson renderer for Salt

See the hjson_ documentation for more information

.. _hjson: http://laktak.github.io/hjson/
"""
try:
    import hjson
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

def render(hjson_data, saltenv='base', sls='', **kws):
    if False:
        print('Hello World!')
    '\n    Accepts HJSON as a string or as a file object and runs it through the HJSON\n    parser.\n\n    :rtype: A Python data structure\n    '
    if not isinstance(hjson_data, str):
        hjson_data = hjson_data.read()
    if hjson_data.startswith('#!'):
        hjson_data = hjson_data[hjson_data.find('\n') + 1:]
    if not hjson_data.strip():
        return {}
    return hjson.loads(hjson_data)