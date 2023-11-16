import json
import re
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request
from wagtail.embeds.exceptions import EmbedException, EmbedNotFoundException
from .base import EmbedFinder

class AccessDeniedFacebookOEmbedException(EmbedException):
    pass

class FacebookOEmbedFinder(EmbedFinder):
    """
    An embed finder that supports the authenticated Facebook oEmbed Endpoint.
    https://developers.facebook.com/docs/plugins/oembed
    """
    facebook_video = {'endpoint': 'https://graph.facebook.com/v11.0/oembed_video', 'urls': ['^https://(?:www\\.)?facebook\\.com/.+?/videos/.+$', '^https://(?:www\\.)?facebook\\.com/video\\.php\\?(?:v|id)=.+$', '^https://fb.watch/.+$']}
    facebook_post = {'endpoint': 'https://graph.facebook.com/v11.0/oembed_post', 'urls': ['^https://(?:www\\.)?facebook\\.com/.+?/(?:posts|activity)/.+$', '^https://(?:www\\.)?facebook\\.com/photo\\.php\\?fbid=.+$', '^https://(?:www\\.)?facebook\\.com/(?:photos|questions)/.+$', '^https://(?:www\\.)?facebook\\.com/permalink\\.php\\?story_fbid=.+$', '^https://(?:www\\.)?facebook\\.com/media/set/?\\?set=.+$', '^https://(?:www\\.)?facebook\\.com/notes/.+?/.+?/.+$', '^https://(?:www\\.)?facebook\\.com/.+?/photos/.+$']}

    def __init__(self, omitscript=False, app_id=None, app_secret=None):
        if False:
            for i in range(10):
                print('nop')
        self.app_id = app_id
        self.app_secret = app_secret
        self.omitscript = omitscript
        self._endpoints = {}
        for provider in [self.facebook_video, self.facebook_post]:
            patterns = []
            endpoint = provider['endpoint'].replace('{format}', 'json')
            for url in provider['urls']:
                patterns.append(re.compile(url))
            self._endpoints[endpoint] = patterns

    def _get_endpoint(self, url):
        if False:
            for i in range(10):
                print('nop')
        for (endpoint, patterns) in self._endpoints.items():
            for pattern in patterns:
                if re.match(pattern, url):
                    return endpoint

    def accept(self, url):
        if False:
            print('Hello World!')
        return self._get_endpoint(url) is not None

    def find_embed(self, url, max_width=None, max_height=None):
        if False:
            for i in range(10):
                print('nop')
        endpoint = self._get_endpoint(url)
        if endpoint is None:
            raise EmbedNotFoundException
        params = {'url': url, 'format': 'json'}
        if max_width:
            params['maxwidth'] = max_width
        if max_height:
            params['maxheight'] = max_height
        if self.omitscript:
            params['omitscript'] = 'true'
        request = Request(endpoint + '?' + urlencode(params))
        request.add_header('Authorization', f'Bearer {self.app_id}|{self.app_secret}')
        try:
            r = urllib_request.urlopen(request)
        except (HTTPError, URLError) as e:
            if isinstance(e, HTTPError) and e.code == 404:
                raise EmbedNotFoundException
            elif isinstance(e, HTTPError) and e.code in [400, 401, 403]:
                raise AccessDeniedFacebookOEmbedException
            else:
                raise EmbedNotFoundException
        oembed = json.loads(r.read().decode('utf-8'))
        return {'title': oembed['title'] if 'title' in oembed else '', 'author_name': oembed['author_name'] if 'author_name' in oembed else '', 'provider_name': oembed['provider_name'] if 'provider_name' in oembed else 'Facebook', 'type': oembed['type'], 'thumbnail_url': oembed.get('thumbnail_url'), 'width': oembed.get('width'), 'height': oembed.get('height'), 'html': oembed.get('html')}
embed_finder_class = FacebookOEmbedFinder