from urllib.parse import urlparse
from typing_extensions import override
from zerver.lib.url_preview.types import UrlEmbedData
from .base import BaseParser

class OpenGraphParser(BaseParser):

    @override
    def extract_data(self) -> UrlEmbedData:
        if False:
            while True:
                i = 10
        meta = self._soup.findAll('meta')
        data = UrlEmbedData()
        for tag in meta:
            if not tag.has_attr('property'):
                continue
            if not tag.has_attr('content'):
                continue
            if tag['property'] == 'og:title':
                data.title = tag['content']
            elif tag['property'] == 'og:description':
                data.description = tag['content']
            elif tag['property'] == 'og:image':
                try:
                    urlparse(tag['content'])
                except ValueError:
                    continue
                data.image = tag['content']
        return data