__package__ = 'archivebox.extractors'
from html.parser import HTMLParser
import io
from pathlib import Path
from typing import Optional
from ..config import SAVE_HTMLTOTEXT, TIMEOUT, VERSION
from ..index.schema import Link, ArchiveResult, ArchiveError
from ..logging_util import TimedProgress
from ..system import atomic_write
from ..util import enforce_types, is_static_file
from .title import get_html

class HTMLTextExtractor(HTMLParser):
    TEXT_ATTRS = ['alt', 'cite', 'href', 'label', 'list', 'placeholder', 'title', 'value']
    NOTEXT_TAGS = ['script', 'style', 'template']
    NOTEXT_HREF = ['data:', 'javascript:', '#']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.output = io.StringIO()
        self._tag_stack = []

    def _is_text_attr(self, name, value):
        if False:
            while True:
                i = 10
        if not isinstance(value, str):
            return False
        if name == 'href' and any(map(lambda p: value.startswith(p), self.NOTEXT_HREF)):
            return False
        if name in self.TEXT_ATTRS:
            return True
        return False

    def _parent_tag(self):
        if False:
            print('Hello World!')
        try:
            return self._tag_stack[-1]
        except IndexError:
            return None

    def _in_notext_tag(self):
        if False:
            i = 10
            return i + 15
        return any([t in self._tag_stack for t in self.NOTEXT_TAGS])

    def handle_starttag(self, tag, attrs):
        if False:
            print('Hello World!')
        self._tag_stack.append(tag)
        if self._in_notext_tag():
            return
        for (name, value) in attrs:
            if self._is_text_attr(name, value):
                self.output.write(f'({value.strip()}) ')

    def handle_endtag(self, tag):
        if False:
            while True:
                i = 10
        orig_stack = self._tag_stack.copy()
        try:
            while tag != self._tag_stack.pop():
                pass
            if not self._in_notext_tag() and tag not in self.NOTEXT_TAGS:
                self.output.write(' ')
        except IndexError:
            self._tag_stack = orig_stack

    def handle_data(self, data):
        if False:
            while True:
                i = 10
        if self._in_notext_tag():
            return
        data = data.lstrip()
        len_before_rstrip = len(data)
        data = data.rstrip()
        spaces_rstripped = len_before_rstrip - len(data)
        if data:
            self.output.write(data)
            if spaces_rstripped:
                self.output.write(' ')

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.output.getvalue()

@enforce_types
def should_save_htmltotext(link: Link, out_dir: Optional[Path]=None, overwrite: Optional[bool]=False) -> bool:
    if False:
        while True:
            i = 10
    if is_static_file(link.url):
        return False
    out_dir = out_dir or Path(link.link_dir)
    if not overwrite and (out_dir / 'htmltotext.txt').exists():
        return False
    return SAVE_HTMLTOTEXT

@enforce_types
def save_htmltotext(link: Link, out_dir: Optional[Path]=None, timeout: int=TIMEOUT) -> ArchiveResult:
    if False:
        print('Hello World!')
    'extract search-indexing-friendly text from an HTML document'
    out_dir = Path(out_dir or link.link_dir)
    output = 'htmltotext.txt'
    timer = TimedProgress(timeout, prefix='      ')
    extracted_text = None
    try:
        extractor = HTMLTextExtractor()
        document = get_html(link, out_dir)
        if not document:
            raise ArchiveError('htmltotext could not find HTML to parse for article text')
        extractor.feed(document)
        extractor.close()
        extracted_text = str(extractor)
        atomic_write(str(out_dir / output), extracted_text)
    except (Exception, OSError) as err:
        status = 'failed'
        output = err
        cmd = ['(internal) archivebox.extractors.htmltotext', './{singlefile,dom}.html']
    finally:
        timer.end()
    return ArchiveResult(cmd=cmd, pwd=str(out_dir), cmd_version=VERSION, output=output, status=status, index_texts=[extracted_text] if extracted_text else [], **timer.stats)