import sys
LOG_NOTSET = 'notset'
LOG_DEBUG = 'debug'
LOG_INFO = 'info'
LOG_WARNING = 'warn'
LOG_ERROR = 'error'
LOG_CRITICAL = 'critical'

def get_encodings(hint_encoding='utf-8'):
    if False:
        return 10
    fallbacks = {'latin1': 'latin9', 'iso-8859-1': 'iso8859-15', 'cp1252': '1252'}
    if hint_encoding:
        yield hint_encoding
        if hint_encoding.lower() in fallbacks:
            yield fallbacks[hint_encoding.lower()]
    for charset in ['utf8', 'latin1']:
        if not hint_encoding or charset.lower() != hint_encoding.lower():
            yield charset
    from locale import getpreferredencoding
    prefenc = getpreferredencoding()
    if prefenc and prefenc.lower() != 'utf-8':
        yield prefenc
        prefenc = fallbacks.get(prefenc.lower())
        if prefenc:
            yield prefenc

def ustr(value, hint_encoding='utf-8', errors='strict'):
    if False:
        return 10
    "This method is similar to the builtin `unicode`, except\n    that it may try multiple encodings to find one that works\n    for decoding `value`, and defaults to 'utf-8' first.\n\n    :param: value: the value to convert\n    :param: hint_encoding: an optional encoding that was detecte\n        upstream and should be tried first to decode ``value``.\n    :param str errors: optional `errors` flag to pass to the unicode\n        built-in to indicate how illegal character values should be\n        treated when converting a string: 'strict', 'ignore' or 'replace'\n        (see ``unicode()`` constructor).\n        Passing anything other than 'strict' means that the first\n        encoding tried will be used, even if it's not the correct\n        one to use, so be careful! Ignored if value is not a string/unicode.\n    :raise: UnicodeError if value cannot be coerced to unicode\n    :return: unicode string representing the given value\n    "
    ttype = type(value)
    if ttype is unicode:
        return value
    if ttype is str or issubclass(ttype, str):
        try:
            return unicode(value, hint_encoding, errors=errors)
        except Exception:
            pass
        for ln in get_encodings(hint_encoding):
            try:
                return unicode(value, ln, errors=errors)
            except Exception:
                pass
    if isinstance(value, Exception):
        return exception_to_unicode(value)
    try:
        return unicode(value)
    except Exception:
        raise UnicodeError('unable to convert %r' % (value,))

def exception_to_unicode(e):
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info[:2] < (2, 6) and hasattr(e, 'message'):
        return ustr(e.message)
    if hasattr(e, 'args'):
        return '\n'.join((ustr(a) for a in e.args))
    try:
        return unicode(e)
    except Exception:
        return u'Unknown message'