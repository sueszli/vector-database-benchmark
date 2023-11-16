"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.data import conf
from lib.core.common import getSafeExString
from lib.core.exception import SqlmapConnectionException
from thirdparty.six.moves import http_client as _http_client
from thirdparty.six.moves import urllib as _urllib

class HTTPSPKIAuthHandler(_urllib.request.HTTPSHandler):

    def __init__(self, auth_file):
        if False:
            for i in range(10):
                print('nop')
        _urllib.request.HTTPSHandler.__init__(self)
        self.auth_file = auth_file

    def https_open(self, req):
        if False:
            print('Hello World!')
        return self.do_open(self.getConnection, req)

    def getConnection(self, host, timeout=None):
        if False:
            i = 10
            return i + 15
        try:
            return _http_client.HTTPSConnection(host, cert_file=self.auth_file, key_file=self.auth_file, timeout=conf.timeout)
        except IOError as ex:
            errMsg = 'error occurred while using key '
            errMsg += "file '%s' ('%s')" % (self.auth_file, getSafeExString(ex))
            raise SqlmapConnectionException(errMsg)