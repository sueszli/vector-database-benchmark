"""Create / interact with Google Cloud Translation connections."""
from google.cloud import _http
from google.cloud.translate_v2 import __version__

class Connection(_http.JSONConnection):
    """A connection to Google Cloud Translation API via the JSON REST API.

    :type client: :class:`~google.cloud.translate.client.Client`
    :param client: The client that owns the current connection.

    :type client_info: :class:`~google.api_core.client_info.ClientInfo`
    :param client_info: (Optional) instance used to generate user agent.
    """
    DEFAULT_API_ENDPOINT = 'https://translation.googleapis.com'

    def __init__(self, client, client_info=None, api_endpoint=DEFAULT_API_ENDPOINT):
        if False:
            while True:
                i = 10
        super(Connection, self).__init__(client, client_info)
        self.API_BASE_URL = api_endpoint
        self._client_info.gapic_version = __version__
        self._client_info.client_library_version = __version__
    API_VERSION = 'v2'
    "The version of the API, used in building the API call's URL."
    API_URL_TEMPLATE = '{api_base_url}/language/translate/{api_version}{path}'
    'A template for the URL of a particular API call.'