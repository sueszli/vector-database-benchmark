"""Convert a NT pathname to a file URL and vice versa.

This module only exists to provide OS-specific code
for urllib.requests, thus do not use directly.
"""

def url2pathname(url):
    if False:
        for i in range(10):
            print('nop')
    "OS-specific conversion from a relative URL of the 'file' scheme\n    to a file system path; not recommended for general use."
    import string, urllib.parse
    url = url.replace(':', '|')
    if not '|' in url:
        if url[:4] == '////':
            url = url[2:]
        components = url.split('/')
        return urllib.parse.unquote('\\'.join(components))
    comp = url.split('|')
    if len(comp) != 2 or comp[0][-1] not in string.ascii_letters:
        error = 'Bad URL: ' + url
        raise OSError(error)
    drive = comp[0][-1].upper()
    components = comp[1].split('/')
    path = drive + ':'
    for comp in components:
        if comp:
            path = path + '\\' + urllib.parse.unquote(comp)
    if path.endswith(':') and url.endswith('/'):
        path += '\\'
    return path

def pathname2url(p):
    if False:
        for i in range(10):
            print('nop')
    "OS-specific conversion from a file system path to a relative URL\n    of the 'file' scheme; not recommended for general use."
    import urllib.parse
    if p[:4] == '\\\\?\\':
        p = p[4:]
        if p[:4].upper() == 'UNC\\':
            p = '\\' + p[4:]
        elif p[1:2] != ':':
            raise OSError('Bad path: ' + p)
    if not ':' in p:
        if p[:2] == '\\\\':
            p = '\\\\' + p
        components = p.split('\\')
        return urllib.parse.quote('/'.join(components))
    comp = p.split(':', maxsplit=2)
    if len(comp) != 2 or len(comp[0]) > 1:
        error = 'Bad path: ' + p
        raise OSError(error)
    drive = urllib.parse.quote(comp[0].upper())
    components = comp[1].split('\\')
    path = '///' + drive + ':'
    for comp in components:
        if comp:
            path = path + '/' + urllib.parse.quote(comp)
    return path