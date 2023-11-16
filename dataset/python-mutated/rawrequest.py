from lib.core.exceptions import InvalidRawRequest
from lib.core.logger import logger
from lib.parse.headers import HeadersParser
from lib.utils.file import File

def parse_raw(raw_file):
    if False:
        while True:
            i = 10
    with File(raw_file) as fd:
        raw_content = fd.read()
    try:
        (head, body) = raw_content.split('\n\n', 1)
    except ValueError:
        try:
            (head, body) = raw_content.split('\r\n\r\n', 1)
        except ValueError:
            head = raw_content.strip('\n')
            body = None
    try:
        (method, path) = head.splitlines()[0].split()[:2]
        headers = HeadersParser('\n'.join(head.splitlines()[1:]))
        host = headers.get('host')
    except KeyError:
        raise InvalidRawRequest("Can't find the Host header in the raw request")
    except Exception as e:
        logger.exception(e)
        raise InvalidRawRequest('The raw request is formatively invalid')
    return ([host + path], method, dict(headers), body)