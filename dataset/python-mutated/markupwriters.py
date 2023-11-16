from .markuputils import attribute_escape, html_escape, xml_escape
from .robottypes import is_string, is_pathlike
from .robotio import file_writer

class _MarkupWriter:

    def __init__(self, output, write_empty=True, usage=None):
        if False:
            while True:
                i = 10
        '\n        :param output: Either an opened, file like object, or a path to the\n            desired output file. In the latter case, the file is created\n            and clients should use :py:meth:`close` method to close it.\n        :param write_empty: Whether to write empty elements and attributes.\n        '
        if is_string(output) or is_pathlike(output):
            output = file_writer(output, usage=usage)
        self.output = output
        self._write_empty = write_empty
        self._preamble()

    def _preamble(self):
        if False:
            print('Hello World!')
        pass

    def start(self, name, attrs=None, newline=True, write_empty=None):
        if False:
            i = 10
            return i + 15
        attrs = self._format_attrs(attrs, write_empty)
        self._start(name, attrs, newline)

    def _start(self, name, attrs, newline):
        if False:
            while True:
                i = 10
        self._write(f'<{name} {attrs}>' if attrs else f'<{name}>', newline)

    def _format_attrs(self, attrs, write_empty):
        if False:
            i = 10
            return i + 15
        if not attrs:
            return ''
        if write_empty is None:
            write_empty = self._write_empty
        return ' '.join((f'''{name}="{attribute_escape(value or '')}"''' for (name, value) in self._order_attrs(attrs) if write_empty or value))

    def _order_attrs(self, attrs):
        if False:
            print('Hello World!')
        return attrs.items()

    def content(self, content=None, escape=True, newline=False):
        if False:
            print('Hello World!')
        if content:
            self._write(self._escape(content) if escape else content, newline)

    def _escape(self, content):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def end(self, name, newline=True):
        if False:
            while True:
                i = 10
        self._write(f'</{name}>', newline)

    def element(self, name, content=None, attrs=None, escape=True, newline=True, write_empty=None):
        if False:
            while True:
                i = 10
        attrs = self._format_attrs(attrs, write_empty)
        if write_empty is None:
            write_empty = self._write_empty
        if write_empty or content or attrs:
            self._start(name, attrs, newline=False)
            self.content(content, escape)
            self.end(name, newline)

    def close(self):
        if False:
            print('Hello World!')
        'Closes the underlying output file.'
        self.output.close()

    def _write(self, text, newline=False):
        if False:
            i = 10
            return i + 15
        self.output.write(text)
        if newline:
            self.output.write('\n')

class HtmlWriter(_MarkupWriter):

    def _order_attrs(self, attrs):
        if False:
            return 10
        return sorted(attrs.items())

    def _escape(self, content):
        if False:
            return 10
        return html_escape(content)

class XmlWriter(_MarkupWriter):

    def _preamble(self):
        if False:
            i = 10
            return i + 15
        self._write('<?xml version="1.0" encoding="UTF-8"?>', newline=True)

    def _escape(self, text):
        if False:
            for i in range(10):
                print('nop')
        return xml_escape(text)

    def element(self, name, content=None, attrs=None, escape=True, newline=True, write_empty=None):
        if False:
            i = 10
            return i + 15
        if content:
            super().element(name, content, attrs, escape, newline, write_empty)
        else:
            self._self_closing_element(name, attrs, newline, write_empty)

    def _self_closing_element(self, name, attrs, newline, write_empty):
        if False:
            while True:
                i = 10
        attrs = self._format_attrs(attrs, write_empty)
        if write_empty is None:
            write_empty = self._write_empty
        if write_empty or attrs:
            self._write(f'<{name} {attrs}/>' if attrs else f'<{name}/>', newline)

class NullMarkupWriter:
    """Null implementation of the _MarkupWriter interface."""
    __init__ = start = content = element = end = close = lambda *args, **kwargs: None