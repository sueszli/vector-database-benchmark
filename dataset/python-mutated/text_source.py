import re
from visidata import vd, BaseSheet
vd.option('regex_skip', '', 'regex of lines to skip in text sources', help='regex')

class FilterFile:

    def __init__(self, fp, regex: str, regex_flags: int=0):
        if False:
            i = 10
            return i + 15
        import re
        self._fp = fp
        self._regex_skip = re.compile(regex, regex_flags)

    def readline(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        while True:
            line = self._fp.readline()
            if self._regex_skip.match(line):
                continue
            return line

    def __getattr__(self, k):
        if False:
            while True:
                i = 10
        return getattr(self._fp, k)

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            return 10
        return self._fp.__exit__(*args, **kwargs)

@BaseSheet.api
def open_text_source(sheet):
    if False:
        return 10
    'Open sheet source as text, using sheet options for encoding and regex_skip.'
    fp = sheet.source.open(encoding=sheet.options.encoding, encoding_errors=sheet.options.encoding_errors)
    regex_skip = sheet.options.regex_skip
    if regex_skip:
        return FilterFile(fp, regex_skip, sheet.regex_flags())
    return fp