"""
$description TV network with multiple live TV channels broadcasting across various Latin American countries.
$url antena7.com.do
$url atv.pe
$url c9n.com.py
$url canal10.com.ni
$url canal12.com.sv
$url chapintv.com
$url elnueve.com.ar
$url redbolivision.tv.bo
$url repretel.com
$url rts.com.ec
$url snt.com.py
$url tvc.com.ec
$url vtv.com.hn
$type live
$region various
"""
import logging
import re
import time
from streamlink.plugin import Plugin, pluginmatcher
from streamlink.plugin.api import validate
from streamlink.stream.hls import HLSStream
from streamlink.utils.url import update_qsd
log = logging.getLogger(__name__)

@pluginmatcher(re.compile('\n    https?://(?:www\\.)?\n    (\n        antena7\\.com\\.do\n        |\n        atv\\.pe\n        |\n        c9n\\.com\\.py\n        |\n        canal10\\.com\\.ni\n        |\n        canal12\\.com\\.sv\n        |\n        chapintv\\.com\n        |\n        elnueve\\.com\\.ar\n        |\n        redbolivision\\.tv\\.bo\n        |\n        repretel\\.com\n        |\n        rts\\.com\\.ec\n        |\n        snt\\.com\\.py\n        |\n        tvc\\.com\\.ec\n        |\n        vtv\\.com\\.hn\n    )\n    /\n    (?:\n        (?:\n            en-?vivo(?:-atv(?:mas)?|-canal-?\\d{1,2})?\n        )\n        |\n        upptv\n    )\n    (?:/|\\#)?$\n', re.VERBOSE))
class Albavision(Plugin):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._page = None

    @property
    def page(self):
        if False:
            print('Hello World!')
        if self._page is None:
            self._page = self.session.http.get(self.url, schema=validate.Schema(validate.parse_html()))
        return self._page

    def _is_token_based_site(self):
        if False:
            i = 10
            return i + 15
        schema = validate.Schema(validate.xml_xpath_string(".//script[contains(text(), 'jQuery.get')]/text()"))
        is_token_based_site = schema.validate(self.page) is not None
        log.debug(f'is_token_based_site={is_token_based_site}')
        return is_token_based_site

    def _get_live_url(self):
        if False:
            while True:
                i = 10
        schema = validate.Schema(validate.xml_xpath_string(".//script[contains(text(), 'LIVE_URL')]/text()"), validate.none_or_all(re.compile('LIVE_URL\\s*=\\s*(?P<q>[\'"])(?P<url>.+?)(?P=q)'), validate.none_or_all(validate.get('url'), validate.url())))
        live_url = schema.validate(self.page)
        log.debug(f'live_url={live_url}')
        return live_url

    def _get_token_req_url(self):
        if False:
            return 10
        schema = validate.Schema(validate.xml_xpath_string(".//script[contains(text(), 'LIVE_URL')]/text()"), validate.none_or_all(re.compile('jQuery\\.get\\s*\\((?P<q>[\'"])(?P<token>.+?)(?P=q)'), validate.none_or_all(validate.get('token'), validate.url())))
        token_req_host = schema.validate(self.page)
        log.debug(f'token_req_host={token_req_host}')
        schema = validate.Schema(validate.xml_xpath_string(".//script[contains(text(), 'LIVE_URL')]/text()"), validate.none_or_all(re.compile('Math\\.floor\\(Date\\.now\\(\\)\\s*/\\s*3600000\\),\\s*(?P<q>[\'"])(?P<token>.+?)(?P=q)'), validate.none_or_all(validate.get('token'))))
        token_req_str = schema.validate(self.page)
        log.debug(f'token_req_str={token_req_str}')
        if not token_req_str:
            return
        date = int(time.time() // 3600)
        token_req_token = self.transform_token(token_req_str, date) or self.transform_token(token_req_str, date - 1)
        if token_req_host and token_req_token:
            return update_qsd(token_req_host, {'rsk': token_req_token})

    def _get_token(self):
        if False:
            while True:
                i = 10
        if not self._is_token_based_site():
            return
        token_req_url = self._get_token_req_url()
        if not token_req_url:
            return
        res = self.session.http.get(token_req_url, schema=validate.Schema(validate.parse_json(), {'success': bool, validate.optional('error'): int, validate.optional('token'): str}))
        if not res['success']:
            if res['error']:
                log.error(f"Token request failed with error: {res['error']}")
            else:
                log.error('Token request failed')
            return
        if not res['token']:
            log.error('Token not found in response')
            return
        token = res['token']
        log.debug(f'token={token}')
        return token

    @staticmethod
    def transform_token(token_in, date):
        if False:
            while True:
                i = 10
        token_out = list(token_in)
        offset = len(token_in)
        for i in range(offset - 1, -1, -1):
            p = i * date % offset
            (token_out[i], token_out[p]) = (token_out[p], token_out[i])
        token_out = ''.join(token_out)
        if token_out.endswith('OK'):
            return token_out[:-2]
        else:
            log.error(f'Invalid site token: {token_in} => {token_out}')

    def _get_streams(self):
        if False:
            for i in range(10):
                print('nop')
        live_url = self._get_live_url()
        if not live_url:
            log.info('This stream may be off-air or not available in your country')
            return
        token = self._get_token()
        if token:
            live_url = update_qsd(live_url, {'iut': token})
        return HLSStream.parse_variant_playlist(self.session, live_url)
__plugin__ = Albavision