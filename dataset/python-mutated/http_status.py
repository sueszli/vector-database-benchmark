"""HTTPStatus exception class."""
from falcon.util import http_status_to_code
from falcon.util.deprecation import AttributeRemovedError

class HTTPStatus(Exception):
    """Represents a generic HTTP status.

    Raise an instance of this class from a hook, middleware, or
    responder to short-circuit request processing in a manner similar
    to ``falcon.HTTPError``, but for non-error status codes.

    Args:
        status (Union[str,int]): HTTP status code or line (e.g.,
            ``'400 Bad Request'``). This may be set to a member of
            :class:`http.HTTPStatus`, an HTTP status line string or byte
            string (e.g., ``'200 OK'``), or an ``int``.
        headers (dict): Extra headers to add to the response.
        text (str): String representing response content. Falcon will encode
            this value as UTF-8 in the response.

    Attributes:
        status (Union[str,int]): The HTTP status line or integer code for
            the status that this exception represents.
        status_code (int): HTTP status code normalized from :attr:`status`.
        headers (dict): Extra headers to add to the response.
        text (str): String representing response content. Falcon will encode
            this value as UTF-8 in the response.

    """
    __slots__ = ('status', 'headers', 'text')

    def __init__(self, status, headers=None, text=None):
        if False:
            print('Hello World!')
        self.status = status
        self.headers = headers
        self.text = text

    @property
    def status_code(self) -> int:
        if False:
            print('Hello World!')
        return http_status_to_code(self.status)

    @property
    def body(self):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeRemovedError('The body attribute is no longer supported. Please use the text attribute instead.')