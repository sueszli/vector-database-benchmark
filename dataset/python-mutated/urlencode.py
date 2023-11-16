"""Bookmarklet/escaped script unpacker."""
try:
    from urllib import unquote_plus
except ImportError:
    from urllib.parse import unquote_plus
PRIORITY = 0

def detect(code):
    if False:
        print('Hello World!')
    'Detects if a scriptlet is urlencoded.'
    return ' ' not in code and ('%20' in code or code.count('%') > 3)

def unpack(code):
    if False:
        print('Hello World!')
    'URL decode `code` source string.'
    return unquote_plus(code) if detect(code) else code