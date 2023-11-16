"""HTTP Basic Authentication tool.

This module provides a CherryPy 3.x tool which implements
the server-side of HTTP Basic Access Authentication, as described in
:rfc:`2617`.

Example usage, using the built-in checkpassword_dict function which uses a dict
as the credentials store::

    userpassdict = {'bird' : 'bebop', 'ornette' : 'wayout'}
    checkpassword = cherrypy.lib.auth_basic.checkpassword_dict(userpassdict)
    basic_auth = {'tools.auth_basic.on': True,
                  'tools.auth_basic.realm': 'earth',
                  'tools.auth_basic.checkpassword': checkpassword,
                  'tools.auth_basic.accept_charset': 'UTF-8',
    }
    app_config = { '/' : basic_auth }

"""
import binascii
import unicodedata
import base64
import cherrypy
from cherrypy._cpcompat import ntou, tonative
__author__ = 'visteya'
__date__ = 'April 2009'

def checkpassword_dict(user_password_dict):
    if False:
        return 10
    'Returns a checkpassword function which checks credentials\n    against a dictionary of the form: {username : password}.\n\n    If you want a simple dictionary-based authentication scheme, use\n    checkpassword_dict(my_credentials_dict) as the value for the\n    checkpassword argument to basic_auth().\n    '

    def checkpassword(realm, user, password):
        if False:
            return 10
        p = user_password_dict.get(user)
        return p and p == password or False
    return checkpassword

def basic_auth(realm, checkpassword, debug=False, accept_charset='utf-8'):
    if False:
        for i in range(10):
            print('nop')
    "A CherryPy tool which hooks at before_handler to perform\n    HTTP Basic Access Authentication, as specified in :rfc:`2617`\n    and :rfc:`7617`.\n\n    If the request has an 'authorization' header with a 'Basic' scheme, this\n    tool attempts to authenticate the credentials supplied in that header.  If\n    the request has no 'authorization' header, or if it does but the scheme is\n    not 'Basic', or if authentication fails, the tool sends a 401 response with\n    a 'WWW-Authenticate' Basic header.\n\n    realm\n        A string containing the authentication realm.\n\n    checkpassword\n        A callable which checks the authentication credentials.\n        Its signature is checkpassword(realm, username, password). where\n        username and password are the values obtained from the request's\n        'authorization' header.  If authentication succeeds, checkpassword\n        returns True, else it returns False.\n\n    "
    fallback_charset = 'ISO-8859-1'
    if '"' in realm:
        raise ValueError('Realm cannot contain the " (quote) character.')
    request = cherrypy.serving.request
    auth_header = request.headers.get('authorization')
    if auth_header is not None:
        msg = 'Bad Request'
        with cherrypy.HTTPError.handle((ValueError, binascii.Error), 400, msg):
            (scheme, params) = auth_header.split(' ', 1)
            if scheme.lower() == 'basic':
                charsets = (accept_charset, fallback_charset)
                decoded_params = base64.b64decode(params.encode('ascii'))
                decoded_params = _try_decode(decoded_params, charsets)
                decoded_params = ntou(decoded_params)
                decoded_params = unicodedata.normalize('NFC', decoded_params)
                decoded_params = tonative(decoded_params)
                (username, password) = decoded_params.split(':', 1)
                if checkpassword(realm, username, password):
                    if debug:
                        cherrypy.log('Auth succeeded', 'TOOLS.AUTH_BASIC')
                    request.login = username
                    return
    charset = accept_charset.upper()
    charset_declaration = ', charset="%s"' % charset if charset != fallback_charset else ''
    cherrypy.serving.response.headers['www-authenticate'] = 'Basic realm="%s"%s' % (realm, charset_declaration)
    raise cherrypy.HTTPError(401, 'You are not authorized to access that resource')

def _try_decode(subject, charsets):
    if False:
        i = 10
        return i + 15
    for charset in charsets[:-1]:
        try:
            return tonative(subject, charset)
        except ValueError:
            pass
    return tonative(subject, charsets[-1])