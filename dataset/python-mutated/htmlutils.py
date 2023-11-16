import re
from urllib.parse import quote
from robot.errors import DataError
from robot.utils import html_escape, html_format, NormalizedDict
from robot.utils.htmlformatters import HeaderFormatter

class DocFormatter:
    _header_regexp = re.compile('<h([234])>(.+?)</h\\1>')
    _name_regexp = re.compile('`(.+?)`')

    def __init__(self, keywords, type_info, introduction, doc_format='ROBOT'):
        if False:
            while True:
                i = 10
        self._doc_to_html = DocToHtml(doc_format)
        self._targets = self._get_targets(keywords, introduction, robot_format=doc_format == 'ROBOT')
        self._type_info_targets = self._get_type_info_targets(type_info)

    def _get_targets(self, keywords, introduction, robot_format):
        if False:
            print('Hello World!')
        targets = {'introduction': 'Introduction', 'library introduction': 'Introduction', 'importing': 'Importing', 'library importing': 'Importing', 'keywords': 'Keywords'}
        for kw in keywords:
            targets[kw.name] = kw.name
        if robot_format:
            for header in self._yield_header_targets(introduction):
                targets[header] = header
        return self._escape_and_encode_targets(targets)

    def _get_type_info_targets(self, type_info):
        if False:
            i = 10
            return i + 15
        targets = {info.name: info.name for info in type_info}
        return self._escape_and_encode_targets(targets)

    def _yield_header_targets(self, introduction):
        if False:
            i = 10
            return i + 15
        headers = HeaderFormatter()
        for line in introduction.splitlines():
            match = headers.match(line.strip())
            if match:
                yield match.group(2)

    def _escape_and_encode_targets(self, targets):
        if False:
            for i in range(10):
                print('nop')
        return NormalizedDict(((html_escape(key), self._encode_uri_component(value)) for (key, value) in targets.items()))

    def _encode_uri_component(self, value):
        if False:
            i = 10
            return i + 15
        return quote(value.encode('UTF-8'), safe="-_.!~*'()")

    def html(self, doc, intro=False):
        if False:
            return 10
        doc = self._doc_to_html(doc)
        if intro:
            doc = self._header_regexp.sub('<h\\1 id="\\2">\\2</h\\1>', doc)
        return self._name_regexp.sub(self._link_keywords, doc)

    def _link_keywords(self, match):
        if False:
            print('Hello World!')
        name = match.group(1)
        targets = self._targets
        types = self._type_info_targets
        if name in targets:
            return f'<a href="#{targets[name]}" class="name">{name}</a>'
        elif name in types:
            return f'<a href="#type-{types[name]}" class="name">{name}</a>'
        return f'<span class="name">{name}</span>'

class DocToHtml:

    def __init__(self, doc_format):
        if False:
            print('Hello World!')
        self._formatter = self._get_formatter(doc_format)

    def _get_formatter(self, doc_format):
        if False:
            print('Hello World!')
        try:
            return {'ROBOT': html_format, 'TEXT': self._format_text, 'HTML': lambda doc: doc, 'REST': self._format_rest}[doc_format]
        except KeyError:
            raise DataError(f"Invalid documentation format '{doc_format}'.")

    def _format_text(self, doc):
        if False:
            for i in range(10):
                print('nop')
        return f'<p style="white-space: pre-wrap">{html_escape(doc)}</p>'

    def _format_rest(self, doc):
        if False:
            return 10
        try:
            from docutils.core import publish_parts
        except ImportError:
            raise DataError("reST format requires 'docutils' module to be installed.")
        parts = publish_parts(doc, writer_name='html', settings_overrides={'syntax_highlight': 'short'})
        return parts['html_body']

    def __call__(self, doc):
        if False:
            print('Hello World!')
        return self._formatter(doc)

class HtmlToText:
    html_tags = {'b': '*', 'i': '_', 'strong': '*', 'em': '_', 'code': '``', 'div.*?': ''}
    html_chars = {'<br */?>': '\n', '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'"}

    def get_short_doc_from_html(self, doc):
        if False:
            i = 10
            return i + 15
        match = re.search('<p.*?>(.*?)</?p>', doc, re.DOTALL)
        if match:
            doc = match.group(1)
        doc = self.html_to_plain_text(doc)
        return doc

    def html_to_plain_text(self, doc):
        if False:
            print('Hello World!')
        for (tag, repl) in self.html_tags.items():
            doc = re.sub('<%(tag)s>(.*?)</%(tag)s>' % {'tag': tag}, '%(repl)s\\1%(repl)s' % {'repl': repl}, doc, flags=re.DOTALL)
        for (html, text) in self.html_chars.items():
            doc = re.sub(html, text, doc)
        return doc