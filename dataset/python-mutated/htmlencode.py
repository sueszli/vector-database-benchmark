"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        return 10
    '\n    HTML encode (using code points) all non-alphanumeric characters (e.g. \' -> &#39;)\n\n    >>> tamper("1\' AND SLEEP(5)#")\n    \'1&#39;&#32;AND&#32;SLEEP&#40;5&#41;&#35;\'\n    >>> tamper("1&#39;&#32;AND&#32;SLEEP&#40;5&#41;&#35;")\n    \'1&#39;&#32;AND&#32;SLEEP&#40;5&#41;&#35;\'\n    '
    if payload:
        payload = re.sub('&#(\\d+);', lambda match: chr(int(match.group(1))), payload)
        payload = re.sub('[^\\w]', lambda match: '&#%d;' % ord(match.group(0)), payload)
    return payload