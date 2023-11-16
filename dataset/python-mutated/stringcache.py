from robot.utils import compress_text, html_format

class StringIndex(int):
    pass

class StringCache:
    empty = StringIndex(0)
    _compress_threshold = 80
    _use_compressed_threshold = 1.1

    def __init__(self):
        if False:
            return 10
        self._cache = {('', False): self.empty}

    def add(self, text, html=False):
        if False:
            print('Hello World!')
        if not text:
            return self.empty
        key = (text, html)
        if key not in self._cache:
            self._cache[key] = StringIndex(len(self._cache))
        return self._cache[key]

    def dump(self):
        if False:
            print('Hello World!')
        return tuple((self._encode(text, html) for (text, html) in self._cache))

    def _encode(self, text, html=False):
        if False:
            i = 10
            return i + 15
        if html:
            text = html_format(text)
        if len(text) > self._compress_threshold:
            compressed = compress_text(text)
            if len(compressed) * self._use_compressed_threshold < len(text):
                return compressed
        return '*' + text