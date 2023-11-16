__package__ = 'archivebox.parsers'
from typing import IO, Iterable
from datetime import datetime
from ..index.schema import Link
from ..util import htmldecode, enforce_types, str_between

@enforce_types
def parse_generic_rss_export(rss_file: IO[str], **_kwargs) -> Iterable[Link]:
    if False:
        i = 10
        return i + 15
    'Parse RSS XML-format files into links'
    rss_file.seek(0)
    items = rss_file.read().split('<item>')
    items = items[1:] if items else []
    for item in items:
        trailing_removed = item.split('</item>', 1)[0]
        leading_removed = trailing_removed.split('<item>', 1)[-1].strip()
        rows = leading_removed.split('\n')

        def get_row(key):
            if False:
                while True:
                    i = 10
            return [r for r in rows if r.strip().startswith('<{}>'.format(key))][0]
        url = str_between(get_row('link'), '<link>', '</link>')
        ts_str = str_between(get_row('pubDate'), '<pubDate>', '</pubDate>')
        time = datetime.strptime(ts_str, '%a, %d %b %Y %H:%M:%S %z')
        title = str_between(get_row('title'), '<![CDATA[', ']]').strip()
        yield Link(url=htmldecode(url), timestamp=str(time.timestamp()), title=htmldecode(title) or None, tags=None, sources=[rss_file.name])
KEY = 'rss'
NAME = 'Generic RSS'
PARSER = parse_generic_rss_export