"""
Clipboard Dummy: an internal implementation that does not use the system
clipboard.
"""
__all__ = ('ClipboardDummy',)
from kivy.core.clipboard import ClipboardBase

class ClipboardDummy(ClipboardBase):

    def __init__(self):
        if False:
            return 10
        super(ClipboardDummy, self).__init__()
        self._data = dict()
        self._data['text/plain'] = None
        self._data['application/data'] = None

    def get(self, mimetype='text/plain'):
        if False:
            i = 10
            return i + 15
        return self._data.get(mimetype, None)

    def put(self, data, mimetype='text/plain'):
        if False:
            while True:
                i = 10
        self._data[mimetype] = data

    def get_types(self):
        if False:
            return 10
        return list(self._data.keys())