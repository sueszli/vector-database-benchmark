import time
from datetime import timedelta

class CookieJar:

    def __init__(self, pluginname, account=None):
        if False:
            while True:
                i = 10
        self.cookies = {}
        self.plugin = pluginname
        self.account = account

    def add_cookies(self, clist):
        if False:
            return 10
        for c in clist:
            name = c.split('\t')[5]
            self.cookies[name] = c

    def get_cookies(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.cookies.values())

    def parse_cookie(self, name):
        if False:
            return 10
        if name in self.cookies:
            return self.cookies[name].split('\t')[6]
        else:
            return None

    def get_cookie(self, name):
        if False:
            i = 10
            return i + 15
        return self.parse_cookie(name)

    def set_cookie(self, domain, name, value, path='/', exp=time.time() + timedelta(days=31).total_seconds()):
        if False:
            for i in range(10):
                print('nop')
        self.cookies[name] = f'.{domain}\tTRUE\t{path}\tFALSE\t{exp}\t{name}\t{value}'

    def clear(self):
        if False:
            return 10
        self.cookies = {}