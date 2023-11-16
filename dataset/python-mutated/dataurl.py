import six
from base64 import b64encode, b64decode
from . import utils
from six.moves.urllib.parse import quote, unquote

def encode(data, mime_type='', charset='utf-8', base64=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Encode data to DataURL\n    '
    if isinstance(data, six.text_type):
        data = data.encode(charset)
    else:
        charset = None
    if base64:
        data = utils.text(b64encode(data))
    else:
        data = utils.text(quote(data))
    result = ['data:']
    if mime_type:
        result.append(mime_type)
    if charset:
        result.append(';charset=')
        result.append(charset)
    if base64:
        result.append(';base64')
    result.append(',')
    result.append(data)
    return ''.join(result)

def decode(data_url):
    if False:
        return 10
    '\n    Decode DataURL data\n    '
    (metadata, data) = data_url.rsplit(',', 1)
    (_, metadata) = metadata.split('data:', 1)
    parts = metadata.split(';')
    if parts[-1] == 'base64':
        data = b64decode(data)
    else:
        data = unquote(data)
    for part in parts:
        if part.startswith('charset='):
            data = data.decode(part[8:])
    return data