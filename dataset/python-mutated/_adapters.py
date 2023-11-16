import re
import textwrap
import email.message
from ._text import FoldedCase

class Message(email.message.Message):
    multiple_use_keys = set(map(FoldedCase, ['Classifier', 'Obsoletes-Dist', 'Platform', 'Project-URL', 'Provides-Dist', 'Provides-Extra', 'Requires-Dist', 'Requires-External', 'Supported-Platform', 'Dynamic']))
    '\n    Keys that may be indicated multiple times per PEP 566.\n    '

    def __new__(cls, orig: email.message.Message):
        if False:
            while True:
                i = 10
        res = super().__new__(cls)
        vars(res).update(vars(orig))
        return res

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._headers = self._repair_headers()

    def __iter__(self):
        if False:
            while True:
                i = 10
        return super().__iter__()

    def _repair_headers(self):
        if False:
            print('Hello World!')

        def redent(value):
            if False:
                for i in range(10):
                    print('nop')
            'Correct for RFC822 indentation'
            if not value or '\n' not in value:
                return value
            return textwrap.dedent(' ' * 8 + value)
        headers = [(key, redent(value)) for (key, value) in vars(self)['_headers']]
        if self._payload:
            headers.append(('Description', self.get_payload()))
        return headers

    @property
    def json(self):
        if False:
            print('Hello World!')
        '\n        Convert PackageMetadata to a JSON-compatible format\n        per PEP 0566.\n        '

        def transform(key):
            if False:
                i = 10
                return i + 15
            value = self.get_all(key) if key in self.multiple_use_keys else self[key]
            if key == 'Keywords':
                value = re.split('\\s+', value)
            tk = key.lower().replace('-', '_')
            return (tk, value)
        return dict(map(transform, map(FoldedCase, self)))