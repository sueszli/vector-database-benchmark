__package__ = 'archivebox.parsers'
from typing import IO, Iterable
from datetime import datetime, timezone
from xml.etree import ElementTree
from ..index.schema import Link
from ..util import htmldecode, enforce_types

@enforce_types
def parse_pinboard_rss_export(rss_file: IO[str], **_kwargs) -> Iterable[Link]:
    if False:
        return 10
    'Parse Pinboard RSS feed files into links'
    rss_file.seek(0)
    root = ElementTree.parse(rss_file).getroot()
    items = root.findall('{http://purl.org/rss/1.0/}item')
    for item in items:
        find = lambda p: item.find(p).text.strip() if item.find(p) is not None else None
        url = find('{http://purl.org/rss/1.0/}link')
        tags = find('{http://purl.org/dc/elements/1.1/}subject')
        title = find('{http://purl.org/rss/1.0/}title')
        ts_str = find('{http://purl.org/dc/elements/1.1/}date')
        if url is None:
            continue
        if ts_str and ts_str[-3:-2] == ':':
            ts_str = ts_str[:-3] + ts_str[-2:]
        if ts_str:
            time = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S%z')
        else:
            time = datetime.now(timezone.utc)
        yield Link(url=htmldecode(url), timestamp=str(time.timestamp()), title=htmldecode(title) or None, tags=htmldecode(tags) or None, sources=[rss_file.name])
KEY = 'pinboard_rss'
NAME = 'Pinboard RSS'
PARSER = parse_pinboard_rss_export