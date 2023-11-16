__package__ = 'archivebox.parsers'
from typing import IO, Iterable
from datetime import datetime
from ..index.schema import Link
from ..util import htmldecode, enforce_types, str_between

@enforce_types
def parse_wallabag_atom_export(rss_file: IO[str], **_kwargs) -> Iterable[Link]:
    if False:
        i = 10
        return i + 15
    'Parse Wallabag Atom files into links'
    rss_file.seek(0)
    entries = rss_file.read().split('<entry>')[1:]
    for entry in entries:
        trailing_removed = entry.split('</entry>', 1)[0]
        leading_removed = trailing_removed.strip()
        splits_fixed = leading_removed.replace('"\n              href="', '" href="')
        rows = splits_fixed.split('\n')

        def get_row(prefix):
            if False:
                i = 10
                return i + 15
            return [row.strip() for row in rows if row.strip().startswith('<{}'.format(prefix))][0]
        title = str_between(get_row('title'), '<title><![CDATA[', ']]></title>').strip()
        url_inside_link = str_between(get_row('link rel="via"'), '<link rel="via">', '</link>')
        url_inside_attr = str_between(get_row('link rel="via"'), 'href="', '"/>')
        ts_str = str_between(get_row('published'), '<published>', '</published>')
        time = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S%z')
        try:
            tags = str_between(get_row('category'), 'label="', '" />')
        except Exception:
            tags = None
        yield Link(url=htmldecode(url_inside_attr or url_inside_link), timestamp=str(time.timestamp()), title=htmldecode(title) or None, tags=tags or '', sources=[rss_file.name])
KEY = 'wallabag_atom'
NAME = 'Wallabag Atom'
PARSER = parse_wallabag_atom_export