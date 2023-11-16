import logging
logger = logging.getLogger(__name__)

class StripContentTypeMiddleware:
    """WSGI middleware to strip Content-Type header for GETs."""

    def __init__(self, app):
        if False:
            while True:
                i = 10
        'Create the new middleware.\n\n        Args:\n            app: a flask application\n        '
        self.app = app

    def __call__(self, environ, start_response):
        if False:
            i = 10
            return i + 15
        'Run the middleware and then call the original WSGI application.'
        if environ['REQUEST_METHOD'] == 'GET':
            try:
                del environ['CONTENT_TYPE']
            except KeyError:
                pass
            else:
                logger.debug('Remove header "Content-Type" from GET request')
        return self.app(environ, start_response)