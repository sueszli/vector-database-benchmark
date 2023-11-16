"""Functions to resolve TF-Hub Modules stored in uncompressed folders on GCS."""
import urllib
from tensorflow_hub import resolver
_UNCOMPRESSED_FORMAT_QUERY = ('tf-hub-format', 'uncompressed')

class HttpUncompressedFileResolver(resolver.HttpResolverBase):
    """Resolves HTTP handles by requesting and reading their GCS location."""

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.path_resolver = resolver.PathResolver()

    def __call__(self, handle):
        if False:
            i = 10
            return i + 15
        'Request the gs:// path for the handle and pass it to PathResolver.'
        handle_with_params = self._append_uncompressed_format_query(handle)
        gcs_location = self._request_gcs_location(handle_with_params)
        return self.path_resolver(gcs_location)

    def _append_uncompressed_format_query(self, handle):
        if False:
            while True:
                i = 10
        return self._append_format_query(handle, _UNCOMPRESSED_FORMAT_QUERY)

    def _request_gcs_location(self, handle_with_params):
        if False:
            i = 10
            return i + 15
        'Request ...?tf-hub-format=uncompressed and return the response body.'
        request = urllib.request.Request(handle_with_params)
        gcs_location = self._call_urlopen(request)
        if not gcs_location.startswith('gs://'):
            raise ValueError('Expected server to return a GCS location but received {}'.format(gcs_location))
        return gcs_location

    def _call_urlopen(self, request):
        if False:
            while True:
                i = 10
        "We expect a '303 See other' response.\n\n    Fail on anything else.\n\n    Args:\n      request: Request to the ...?tf-hub-format=uncompressed URL.\n\n    Returns:\n      String containing the server response\n\n    Raise a ValueError if\n    - a HTTPError != 303 occurrs\n    - urlopen does not raise an HTTPError (on 2xx responses)\n    "

        def raise_on_unexpected_code(code):
            if False:
                i = 10
                return i + 15
            raise ValueError('Expected 303 See other HTTP response but received code {}'.format(code))
        try:
            response = super()._call_urlopen(request)
            raise_on_unexpected_code(response.code)
        except urllib.error.HTTPError as error:
            if error.code != 303:
                raise_on_unexpected_code(error.code)
            return error.read().decode()

    def is_supported(self, handle):
        if False:
            while True:
                i = 10
        if not self.is_http_protocol(handle):
            return False
        load_format = resolver.model_load_format()
        return load_format == resolver.ModelLoadFormat.UNCOMPRESSED.value