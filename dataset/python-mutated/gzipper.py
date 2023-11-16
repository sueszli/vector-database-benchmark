import cStringIO
import gzip
import wsgiref.headers
from paste.util.mimeparse import parse_mime_type, desired_matches
ENCODABLE_CONTENT_TYPES = {'application/json', 'application/javascript', 'application/xml', 'text/css', 'text/csv', 'text/html', 'text/javascript', 'text/plain', 'text/xml'}

class GzipMiddleware(object):
    """A middleware that transparently compresses content with gzip.

    Note: this middleware deliberately violates PEP-333 in three ways:

        - it disables the use of the "write()" callable.
        - it does content encoding which is a "hop-by-hop" feature.
        - it does not "yield at least one value each time its underlying
          application yields a value".

    None of these are an issue for the reddit application, but use at your
    own risk.

    """

    def __init__(self, app, compression_level, min_size):
        if False:
            return 10
        self.app = app
        self.compression_level = compression_level
        self.min_size = min_size

    def _start_response(self, status, response_headers, exc_info=None):
        if False:
            for i in range(10):
                print('nop')
        self.status = status
        self.headers = response_headers
        self.exc_info = exc_info
        return self._write_not_implemented

    @staticmethod
    def _write_not_implemented(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Raise an exception.\n\n        This middleware doesn't work with the write callable.\n\n        "
        raise NotImplementedError

    @staticmethod
    def content_length(headers, app_iter):
        if False:
            i = 10
            return i + 15
        "Return the content-length of this response as best as we can tell.\n\n        If the application returned a Content-Length header we will trust it.\n        If not, we are allowed by PEP-333 to attempt to determine the length of\n        the app's iterable and if it's 1, use the length of the only chunk as\n        the content-length.\n\n        "
        content_length_header = headers['Content-Length']
        if content_length_header:
            return int(content_length_header)
        try:
            app_iter_len = len(app_iter)
        except ValueError:
            return None
        if app_iter_len == 1:
            return len(app_iter[0])
        return None

    def should_gzip_response(self, headers, app_iter):
        if False:
            return 10
        if 'ETag' in headers:
            return False
        if 'Content-Encoding' in headers:
            return False
        content_length = self.content_length(headers, app_iter)
        if not content_length or content_length < self.min_size:
            return False
        content_type = headers['Content-Type']
        (type, subtype, params) = parse_mime_type(content_type)
        if '%s/%s' % (type, subtype) not in ENCODABLE_CONTENT_TYPES:
            return False
        return True

    @staticmethod
    def update_vary_header(headers):
        if False:
            return 10
        vary_headers = headers.get_all('Vary')
        del headers['Vary']
        varies = []
        for vary_header in vary_headers:
            varies.extend((field.strip().lower() for field in vary_header.split(',')))
        if '*' in varies:
            varies = ['*']
        elif 'accept-encoding' not in varies:
            varies.append('accept-encoding')
        headers['Vary'] = ', '.join(varies)

    @staticmethod
    def request_accepts_gzip(environ):
        if False:
            print('Hello World!')
        accept_encoding = environ.get('HTTP_ACCEPT_ENCODING', 'identity')
        return 'gzip' in desired_matches(['gzip'], accept_encoding)

    def __call__(self, environ, start_response):
        if False:
            for i in range(10):
                print('nop')
        app_iter = self.app(environ, self._start_response)
        headers = wsgiref.headers.Headers(self.headers)
        response_compressible = self.should_gzip_response(headers, app_iter)
        if response_compressible:
            self.update_vary_header(headers)
        if response_compressible and self.request_accepts_gzip(environ):
            headers['Content-Encoding'] = 'gzip'
            response_buffer = cStringIO.StringIO()
            gzipper = gzip.GzipFile(fileobj=response_buffer, mode='wb', compresslevel=self.compression_level)
            try:
                for chunk in app_iter:
                    gzipper.write(chunk)
            finally:
                if hasattr(app_iter, 'close'):
                    app_iter.close()
            gzipper.close()
            new_response = response_buffer.getvalue()
            encoded_app_iter = [new_response]
            response_buffer.close()
            headers['Content-Length'] = str(len(new_response))
        else:
            encoded_app_iter = app_iter
        start_response(self.status, self.headers, self.exc_info)
        return encoded_app_iter

def make_gzip_middleware(app, global_conf=None, compress_level=9, min_size=0):
    if False:
        print('Hello World!')
    'Return a gzip-compressing middleware.'
    return GzipMiddleware(app, int(compress_level), int(min_size))