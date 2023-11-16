from django.utils.html import format_html
from wagtail.embeds.exceptions import EmbedException, EmbedNotFoundException
from .base import EmbedFinder

class EmbedlyException(EmbedException):
    pass

class AccessDeniedEmbedlyException(EmbedlyException):
    pass

class EmbedlyFinder(EmbedFinder):
    key = None

    def __init__(self, key=None):
        if False:
            print('Hello World!')
        if key:
            self.key = key

    def get_key(self):
        if False:
            while True:
                i = 10
        return self.key

    def accept(self, url):
        if False:
            print('Hello World!')
        return True

    def find_embed(self, url, max_width=None, key=None):
        if False:
            i = 10
            return i + 15
        from embedly import Embedly
        if key is None:
            key = self.get_key()
        client = Embedly(key=key)
        if max_width is not None:
            oembed = client.oembed(url, maxwidth=max_width, better=False)
        else:
            oembed = client.oembed(url, better=False)
        if oembed.get('error'):
            if oembed['error_code'] in [401, 403]:
                raise AccessDeniedEmbedlyException
            elif oembed['error_code'] == 404:
                raise EmbedNotFoundException
            else:
                raise EmbedlyException
        if oembed['type'] == 'photo':
            html = format_html('<img src="{}" alt="">', oembed['url'])
        else:
            html = oembed.get('html')
        return {'title': oembed['title'] if 'title' in oembed else '', 'author_name': oembed['author_name'] if 'author_name' in oembed else '', 'provider_name': oembed['provider_name'] if 'provider_name' in oembed else '', 'type': oembed['type'], 'thumbnail_url': oembed.get('thumbnail_url'), 'width': oembed.get('width'), 'height': oembed.get('height'), 'html': html}
embed_finder_class = EmbedlyFinder