import sys
from pprint import PrettyPrinter
from unicodedata import normalize

def safe_str(item):
    if False:
        for i in range(10):
            print('nop')
    return normalize('NFC', _safe_str(item))

def _safe_str(item):
    if False:
        print('Hello World!')
    if isinstance(item, str):
        return item
    if isinstance(item, (bytes, bytearray)):
        try:
            return item.decode('ASCII')
        except UnicodeError:
            return ''.join((chr(b) if b < 128 else '\\x%x' % b for b in item))
    try:
        return str(item)
    except:
        return _unrepresentable_object(item)

def prepr(item, width=80, sort_dicts=False):
    if False:
        return 10
    return safe_str(PrettyRepr(width=width, sort_dicts=sort_dicts).pformat(item))

class PrettyRepr(PrettyPrinter):

    def format(self, object, context, maxlevels, level):
        if False:
            for i in range(10):
                print('nop')
        try:
            return PrettyPrinter.format(self, object, context, maxlevels, level)
        except:
            return (_unrepresentable_object(object), True, False)

    def _format(self, object, *args, **kwargs):
        if False:
            print('Hello World!')
        if isinstance(object, (str, bytes, bytearray)):
            width = self._width
            self._width = sys.maxsize
            try:
                super()._format(object, *args, **kwargs)
            finally:
                self._width = width
        else:
            super()._format(object, *args, **kwargs)

def _unrepresentable_object(item):
    if False:
        i = 10
        return i + 15
    from .error import get_error_message
    return '<Unrepresentable object %s. Error: %s>' % (item.__class__.__name__, get_error_message())