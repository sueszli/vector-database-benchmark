from geopy.geocoders.base import DEFAULT_SENTINEL
from geopy.geocoders.pelias import Pelias
__all__ = ('GeocodeEarth',)

class GeocodeEarth(Pelias):
    """Geocode Earth, a Pelias-based service provided by the developers
    of Pelias itself.

    Documentation at:
        https://geocode.earth/docs

    Pricing details:
        https://geocode.earth/#pricing
    """

    def __init__(self, api_key, *, domain='api.geocode.earth', timeout=DEFAULT_SENTINEL, proxies=DEFAULT_SENTINEL, user_agent=None, scheme=None, ssl_context=DEFAULT_SENTINEL, adapter_factory=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param str api_key: Geocode.earth API key, required.\n\n        :param str domain: Specify a custom domain for Pelias API.\n\n        :param int timeout:\n            See :attr:`geopy.geocoders.options.default_timeout`.\n\n        :param dict proxies:\n            See :attr:`geopy.geocoders.options.default_proxies`.\n\n        :param str user_agent:\n            See :attr:`geopy.geocoders.options.default_user_agent`.\n\n        :param str scheme:\n            See :attr:`geopy.geocoders.options.default_scheme`.\n\n        :type ssl_context: :class:`ssl.SSLContext`\n        :param ssl_context:\n            See :attr:`geopy.geocoders.options.default_ssl_context`.\n\n        :param callable adapter_factory:\n            See :attr:`geopy.geocoders.options.default_adapter_factory`.\n\n            .. versionadded:: 2.0\n\n        '
        super().__init__(api_key=api_key, domain=domain, timeout=timeout, proxies=proxies, user_agent=user_agent, scheme=scheme, ssl_context=ssl_context, adapter_factory=adapter_factory)