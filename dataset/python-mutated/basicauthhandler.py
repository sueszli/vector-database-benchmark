"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from thirdparty.six.moves import urllib as _urllib

class SmartHTTPBasicAuthHandler(_urllib.request.HTTPBasicAuthHandler):
    """
    Reference: http://selenic.com/hg/rev/6c51a5056020
    Fix for a: http://bugs.python.org/issue8797
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        _urllib.request.HTTPBasicAuthHandler.__init__(self, *args, **kwargs)
        self.retried_req = set()
        self.retried_count = 0

    def reset_retry_count(self):
        if False:
            i = 10
            return i + 15
        pass

    def http_error_auth_reqed(self, auth_header, host, req, headers):
        if False:
            i = 10
            return i + 15
        if hash(req) not in self.retried_req:
            self.retried_req.add(hash(req))
            self.retried_count = 0
        elif self.retried_count > 5:
            raise _urllib.error.HTTPError(req.get_full_url(), 401, 'basic auth failed', headers, None)
        else:
            self.retried_count += 1
        return _urllib.request.HTTPBasicAuthHandler.http_error_auth_reqed(self, auth_header, host, req, headers)