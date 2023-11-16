"""
Welcome to the encoding dance!

In a nutshell, text columns are actually a proxy class for byte columns,
which just encode/decodes contents.
"""
from mitmproxy.tools.console import signals
from mitmproxy.tools.console.grideditor import col_bytes

class Column(col_bytes.Column):

    def __init__(self, heading, encoding='utf8', errors='surrogateescape'):
        if False:
            i = 10
            return i + 15
        super().__init__(heading)
        self.encoding_args = (encoding, errors)

    def Display(self, data):
        if False:
            i = 10
            return i + 15
        return TDisplay(data, self.encoding_args)

    def Edit(self, data):
        if False:
            i = 10
            return i + 15
        return TEdit(data, self.encoding_args)

    def blank(self):
        if False:
            i = 10
            return i + 15
        return ''

class EncodingMixin:

    def __init__(self, data, encoding_args):
        if False:
            return 10
        self.encoding_args = encoding_args
        super().__init__(str(data).encode(*self.encoding_args))

    def get_data(self):
        if False:
            return 10
        data = super().get_data()
        try:
            return data.decode(*self.encoding_args)
        except ValueError:
            signals.status_message.send(message='Invalid encoding.')
            raise

class TDisplay(EncodingMixin, col_bytes.Display):
    pass

class TEdit(EncodingMixin, col_bytes.Edit):
    pass