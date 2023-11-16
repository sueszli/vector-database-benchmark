"""
$description Turkish live TV channels and video on-demand service from Dogan Group, including CNN Turk and Kanal D.
$url cnnturk.com
$url dreamturk.com.tr
$url dreamtv.com.tr
$url kanald.com.tr
$url teve2.com.tr
$type live, vod
"""
import logging
import re
from urllib.parse import urljoin
from streamlink.plugin import Plugin, PluginError, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.hls import HLSStream
log = logging.getLogger(__name__)

@pluginmatcher(re.compile('https?://(?:www\\.)?cnnturk\\.com/'))
@pluginmatcher(re.compile('https?://(?:www\\.)?(dreamturk|dreamtv)\\.com\\.tr/'))
@pluginmatcher(re.compile('https?://(?:www\\.)?teve2\\.com\\.tr/'))
@pluginmatcher(re.compile('https?://(?:www\\.)?kanald\\.com\\.tr/'))
class Dogan(Plugin):
    API_URLS = ['/api/media?id={id}', '/actions/content/media/{id}', '/action/media/{id}']
    API_URL_OLD = '/actions/media?id={id}'

    @staticmethod
    def _get_hls_url(root):
        if False:
            return 10
        schema = validate.Schema(validate.xml_xpath_string(".//*[@data-live][contains(@data-url,'.m3u8')]/@data-url"))
        return schema.validate(root)

    @staticmethod
    def _get_content_id(root):
        if False:
            print('Hello World!')
        schema = validate.Schema(validate.any(validate.all(validate.xml_xpath_string("\n                        .//div[@data-id][\n                            @data-live\n                            or @id='video-element'\n                            or @id='player-container'\n                            or contains(@class, 'player-container')\n                        ][1]/@data-id\n                    "), str), validate.all(validate.xml_xpath_string('.//body[@data-content-id][1]/@data-content-id'), str)))
        return schema.validate(root)

    def _api_query_new(self, content_id, api_url):
        if False:
            for i in range(10):
                print('nop')
        url = urljoin(self.url, api_url.format(id=content_id))
        data = self.session.http.get(url, schema=validate.Schema(validate.parse_json(), validate.any(validate.all(str, validate.parse_json(), {'Error': str}, validate.get('Error')), validate.all({'Media': {'Link': {'ContentId': str, validate.optional('DefaultServiceUrl'): validate.any(validate.url(), ''), validate.optional('ServiceUrl'): validate.any(validate.url(), ''), 'SecurePath': str}}}, validate.get(('Media', 'Link')), validate.union_get('ServiceUrl', 'DefaultServiceUrl', 'SecurePath', 'ContentId')))))
        if isinstance(data, str):
            log.error(data)
            return
        (service_url, default_service_url, secure_path, content_id) = data
        if default_service_url == 'https://www.kanald.com.tr':
            self.url = default_service_url
            return self._api_query_old(content_id)
        if re.match('^https?://', secure_path):
            return secure_path
        return urljoin(service_url or default_service_url, secure_path)

    def _api_query_old(self, content_id):
        if False:
            return 10
        url = urljoin(self.url, self.API_URL_OLD.format(id=content_id))
        (service_url, default_service_url, secure_path) = self.session.http.get(url, schema=validate.Schema(validate.parse_json(), {'data': {'id': str, 'media': {'link': {validate.optional('defaultServiceUrl'): validate.any(validate.url(), ''), validate.optional('serviceUrl'): validate.any(validate.url(), ''), 'securePath': str}}}}, validate.get(('data', 'media', 'link')), validate.union_get('serviceUrl', 'defaultServiceUrl', 'securePath')))
        return urljoin(service_url or default_service_url, secure_path)

    def _query_hls_url(self, content_id):
        if False:
            return 10
        for (idx, match) in enumerate(self.matches[:len(self.API_URLS)]):
            if match:
                return self._api_query_new(content_id, self.API_URLS[idx])
        return self._api_query_old(content_id)

    def _get_streams(self):
        if False:
            while True:
                i = 10
        root = self.session.http.get(self.url, schema=validate.Schema(validate.parse_html()))
        hls_url = self._get_hls_url(root)
        if not hls_url:
            try:
                content_id = self._get_content_id(root)
            except PluginError:
                log.error('Could not find the content ID for this stream')
                return
            log.debug(f'Loading content: {content_id}')
            hls_url = self._query_hls_url(content_id)
        if hls_url:
            return HLSStream.parse_variant_playlist(self.session, hls_url)
__plugin__ = Dogan