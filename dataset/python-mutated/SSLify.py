"""Module to enable SSL for Flask application. Adapted from
https://github.com/kennethreitz/flask-sslify
"""
import os
import ssl
from flask import redirect, request
YEAR_IN_SECS = 31536000

class SSLify(object):
    """Secures your Flask App."""

    def __init__(self, app=None, age=YEAR_IN_SECS, subdomains=False, permanent=False, skips=None):
        if False:
            i = 10
            return i + 15
        self.app = app
        self.hsts_age = age
        self.hsts_include_subdomains = subdomains
        self.permanent = permanent
        self.skip_list = skips
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        if False:
            while True:
                i = 10
        'Configures the specified Flask app to enforce SSL.'
        app.config.setdefault('SSLIFY_SUBDOMAINS', False)
        app.config.setdefault('SSLIFY_PERMANENT', False)
        app.config.setdefault('SSLIFY_SKIPS', None)
        self.hsts_include_subdomains = self.hsts_include_subdomains or app.config['SSLIFY_SUBDOMAINS']
        self.permanent = self.permanent or self.app.config['SSLIFY_PERMANENT']
        self.skip_list = self.skip_list or self.app.config['SSLIFY_SKIPS']
        app.before_request(self.redirect_to_ssl)
        app.after_request(self.set_hsts_header)

    @property
    def hsts_header(self):
        if False:
            return 10
        'Returns the proper HSTS policy.'
        hsts_policy = 'max-age={0}'.format(self.hsts_age)
        if self.hsts_include_subdomains:
            hsts_policy += '; includeSubDomains'
        return hsts_policy

    @property
    def skip(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks the skip list.'
        if self.skip_list and isinstance(self.skip_list, list):
            for skip in self.skip_list:
                if request.path.startswith('/{0}'.format(skip)):
                    return True
        return False

    def redirect_to_ssl(self):
        if False:
            return 10
        'Redirect incoming requests to HTTPS.'
        criteria = [request.is_secure, self.app.debug, self.app.testing, request.headers.get('X-Forwarded-Proto', 'http') == 'https']
        if not any(criteria) and (not self.skip):
            if request.url.startswith('http://'):
                url = request.url.replace('http://', 'https://', 1)
                code = 302
                if self.permanent:
                    code = 301
                r = redirect(url, code=code)
                return r

    def set_hsts_header(self, response):
        if False:
            return 10
        'Adds HSTS header to each response.'
        if request.is_secure and (not self.skip):
            response.headers.setdefault('Strict-Transport-Security', self.hsts_header)
        return response

def get_ssl_context(private_key, certificate):
    if False:
        print('Hello World!')
    'Get ssl context from private key and certificate paths.\n    The return value is used when calling Flask.\n    i.e. app.run(ssl_context=get_ssl_context(,,,))\n    '
    if certificate and os.path.isfile(certificate) and private_key and os.path.isfile(private_key):
        context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        context.load_cert_chain(certificate, private_key)
        return context
    return None