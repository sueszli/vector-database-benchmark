from geopy.geocoders.base import DEFAULT_SENTINEL
from geopy.geocoders.nominatim import Nominatim
__all__ = ('PickPoint',)

class PickPoint(Nominatim):
    """PickPoint geocoder is a commercial version of Nominatim.

    Documentation at:
       https://pickpoint.io/api-reference
    """
    geocode_path = '/v1/forward'
    reverse_path = '/v1/reverse'

    def __init__(self, api_key, *, timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, domain='api.pickpoint.io', scheme=None, user_agent=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            return 10
        '\n\n        :param str api_key: PickPoint API key obtained at\n            https://pickpoint.io.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str domain: Domain where the target Nominatim service\n            is hosted.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n        '
        super().__init__(timeout=timeout, proxies=proxies, domain=domain, scheme=scheme, user_agent=user_agent, ssl_context=ssl_context, adapter_factory=adapter_factory)
        self.api_key = api_key

    def _construct_url(self, base_api, params):
        if False:
            i = 10
            return i + 15
        '\n        Construct geocoding request url. Overridden.\n\n        :param str base_api: Geocoding function base address - self.api\n            or self.reverse_api.\n\n        :param dict params: Geocoding params.\n\n        :return: string URL.\n        '
        params['key'] = self.api_key
        return super()._construct_url(base_api, params)