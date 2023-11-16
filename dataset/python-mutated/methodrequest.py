"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.convert import getText
from thirdparty.six.moves import urllib as _urllib

class MethodRequest(_urllib.request.Request):
    """
    Used to create HEAD/PUT/DELETE/... requests with urllib
    """

    def set_method(self, method):
        if False:
            print('Hello World!')
        self.method = getText(method.upper())

    def get_method(self):
        if False:
            return 10
        return getattr(self, 'method', _urllib.request.Request.get_method(self))