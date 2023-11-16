import re
from functools import partial
from itertools import cycle

class LinkFormatter:
    _image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg')
    _link = re.compile('\\[(.+?\\|.*?)\\]')
    _url = re.compile('\n((^|\\ ) ["\'(\\[{]*)           # begin of line or space and opt. any char "\'([{\n([a-z][\\w+-.]*://[^\\s|]+?)   # url\n(?=[)\\]}"\'.,!?:;|]* ($|\\ ))  # opt. any char )]}"\'.,!?:;| and eol or space\n', re.VERBOSE | re.MULTILINE | re.IGNORECASE)

    def format_url(self, text):
        if False:
            while True:
                i = 10
        return self._format_url(text, format_as_image=False)

    def _format_url(self, text, format_as_image=True):
        if False:
            print('Hello World!')
        if '://' not in text:
            return text
        return self._url.sub(partial(self._replace_url, format_as_image), text)

    def _replace_url(self, format_as_image, match):
        if False:
            for i in range(10):
                print('nop')
        pre = match.group(1)
        url = match.group(3)
        if format_as_image and self._is_image(url):
            return pre + self._get_image(url)
        return pre + self._get_link(url)

    def _get_image(self, src, title=None):
        if False:
            i = 10
            return i + 15
        return '<img src="%s" title="%s">' % (self._quot(src), self._quot(title or src))

    def _get_link(self, href, content=None):
        if False:
            for i in range(10):
                print('nop')
        return '<a href="%s">%s</a>' % (self._quot(href), content or href)

    def _quot(self, attr):
        if False:
            i = 10
            return i + 15
        return attr if '"' not in attr else attr.replace('"', '&quot;')

    def format_link(self, text):
        if False:
            while True:
                i = 10
        tokens = self._link.split(text)
        formatters = cycle((self._format_url, self._format_link))
        return ''.join((f(t) for (f, t) in zip(formatters, tokens)))

    def _format_link(self, text):
        if False:
            return 10
        (link, content) = [t.strip() for t in text.split('|', 1)]
        if self._is_image(content):
            content = self._get_image(content, link)
        elif self._is_image(link):
            return self._get_image(link, content)
        return self._get_link(link, content)

    def _is_image(self, text):
        if False:
            print('Hello World!')
        return text.startswith('data:image/') or text.lower().endswith(self._image_exts)

class LineFormatter:
    handles = lambda self, line: True
    newline = '\n'
    _bold = re.compile('\n(                         # prefix (group 1)\n  (^|\\ )                  # begin of line or space\n  ["\'(]* _?               # optionally any char "\'( and optional begin of italic\n)                         #\n\\*                        # start of bold\n([^\\ ].*?)                # no space and then anything (group 3)\n\\*                        # end of bold\n(?=                       # start of postfix (non-capturing group)\n  _? ["\').,!?:;]*         # optional end of italic and any char "\').,!?:;\n  ($|\\ )                  # end of line or space\n)\n', re.VERBOSE)
    _italic = re.compile('\n( (^|\\ ) ["\'(]* )          # begin of line or space and opt. any char "\'(\n_                          # start of italic\n([^\\ _].*?)                # no space or underline and then anything\n_                          # end of italic\n(?= ["\').,!?:;]* ($|\\ ) )  # opt. any char "\').,!?:; and end of line or space\n', re.VERBOSE)
    _code = re.compile('\n( (^|\\ ) ["\'(]* )          # same as above with _ changed to ``\n``\n([^\\ `].*?)\n``\n(?= ["\').,!?:;]* ($|\\ ) )\n', re.VERBOSE)

    def __init__(self):
        if False:
            print('Hello World!')
        self._formatters = [('*', self._format_bold), ('_', self._format_italic), ('``', self._format_code), ('', LinkFormatter().format_link)]

    def format(self, line):
        if False:
            return 10
        for (marker, formatter) in self._formatters:
            if marker in line:
                line = formatter(line)
        return line

    def _format_bold(self, line):
        if False:
            print('Hello World!')
        return self._bold.sub('\\1<b>\\3</b>', line)

    def _format_italic(self, line):
        if False:
            while True:
                i = 10
        return self._italic.sub('\\1<i>\\3</i>', line)

    def _format_code(self, line):
        if False:
            while True:
                i = 10
        return self._code.sub('\\1<code>\\3</code>', line)

class HtmlFormatter:

    def __init__(self):
        if False:
            return 10
        self._formatters = [TableFormatter(), PreformattedFormatter(), ListFormatter(), HeaderFormatter(), RulerFormatter()]
        self._formatters.append(ParagraphFormatter(self._formatters[:]))
        self._current = None

    def format(self, text):
        if False:
            for i in range(10):
                print('nop')
        results = []
        for line in text.splitlines():
            self._process_line(line, results)
        self._end_current(results)
        return '\n'.join(results)

    def _process_line(self, line, results):
        if False:
            return 10
        if not line.strip():
            self._end_current(results)
        elif self._current and self._current.handles(line):
            self._current.add(line)
        else:
            self._end_current(results)
            self._current = self._find_formatter(line)
            self._current.add(line)

    def _end_current(self, results):
        if False:
            for i in range(10):
                print('nop')
        if self._current:
            results.append(self._current.end())
            self._current = None

    def _find_formatter(self, line):
        if False:
            while True:
                i = 10
        for formatter in self._formatters:
            if formatter.handles(line):
                return formatter

class _Formatter:
    _strip_lines = True

    def __init__(self):
        if False:
            print('Hello World!')
        self._lines = []

    def handles(self, line):
        if False:
            while True:
                i = 10
        return self._handles(line.strip() if self._strip_lines else line)

    def _handles(self, line):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def add(self, line):
        if False:
            for i in range(10):
                print('nop')
        self._lines.append(line.strip() if self._strip_lines else line)

    def end(self):
        if False:
            while True:
                i = 10
        result = self.format(self._lines)
        self._lines = []
        return result

    def format(self, lines):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class _SingleLineFormatter(_Formatter):

    def _handles(self, line):
        if False:
            print('Hello World!')
        return not self._lines and self.match(line)

    def match(self, line):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def format(self, lines):
        if False:
            for i in range(10):
                print('nop')
        return self.format_line(lines[0])

    def format_line(self, line):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class RulerFormatter(_SingleLineFormatter):
    match = re.compile('^-{3,}$').match

    def format_line(self, line):
        if False:
            i = 10
            return i + 15
        return '<hr>'

class HeaderFormatter(_SingleLineFormatter):
    match = re.compile('^(={1,3})\\s+(\\S.*?)\\s+\\1$').match

    def format_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        (level, text) = self.match(line).groups()
        level = len(level) + 1
        return '<h%d>%s</h%d>' % (level, text, level)

class ParagraphFormatter(_Formatter):
    _format_line = LineFormatter().format

    def __init__(self, other_formatters):
        if False:
            i = 10
            return i + 15
        _Formatter.__init__(self)
        self._other_formatters = other_formatters

    def _handles(self, line):
        if False:
            return 10
        return not any((other.handles(line) for other in self._other_formatters))

    def format(self, lines):
        if False:
            print('Hello World!')
        return '<p>%s</p>' % self._format_line(' '.join(lines))

class TableFormatter(_Formatter):
    _table_line = re.compile('^\\| (.* |)\\|$')
    _line_splitter = re.compile(' \\|(?= )')
    _format_cell_content = LineFormatter().format

    def _handles(self, line):
        if False:
            print('Hello World!')
        return self._table_line.match(line) is not None

    def format(self, lines):
        if False:
            return 10
        return self._format_table([self._split_to_cells(l) for l in lines])

    def _split_to_cells(self, line):
        if False:
            print('Hello World!')
        return [cell.strip() for cell in self._line_splitter.split(line[1:-1])]

    def _format_table(self, rows):
        if False:
            while True:
                i = 10
        maxlen = max((len(row) for row in rows))
        table = ['<table border="1">']
        for row in rows:
            row += [''] * (maxlen - len(row))
            table.append('<tr>')
            table.extend((self._format_cell(cell) for cell in row))
            table.append('</tr>')
        table.append('</table>')
        return '\n'.join(table)

    def _format_cell(self, content):
        if False:
            i = 10
            return i + 15
        if content.startswith('=') and content.endswith('='):
            tx = 'th'
            content = content[1:-1].strip()
        else:
            tx = 'td'
        return '<%s>%s</%s>' % (tx, self._format_cell_content(content), tx)

class PreformattedFormatter(_Formatter):
    _format_line = LineFormatter().format

    def _handles(self, line):
        if False:
            for i in range(10):
                print('nop')
        return line.startswith('| ') or line == '|'

    def format(self, lines):
        if False:
            for i in range(10):
                print('nop')
        lines = [self._format_line(line[2:]) for line in lines]
        return '\n'.join(['<pre>'] + lines + ['</pre>'])

class ListFormatter(_Formatter):
    _strip_lines = False
    _format_item = LineFormatter().format

    def _handles(self, line):
        if False:
            i = 10
            return i + 15
        return line.strip().startswith('- ') or (line.startswith(' ') and self._lines)

    def format(self, lines):
        if False:
            for i in range(10):
                print('nop')
        items = ['<li>%s</li>' % self._format_item(line) for line in self._combine_lines(lines)]
        return '\n'.join(['<ul>'] + items + ['</ul>'])

    def _combine_lines(self, lines):
        if False:
            i = 10
            return i + 15
        current = []
        for line in lines:
            line = line.strip()
            if not line.startswith('- '):
                current.append(line)
                continue
            if current:
                yield ' '.join(current)
            current = [line[2:].strip()]
        yield ' '.join(current)