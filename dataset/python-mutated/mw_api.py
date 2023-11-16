import time
import hashlib
import sys
import os
import json

class mwApi:
    __MW_KEY = 'uATE5NrKDWIlZuDcYpvLVhoUo1c7A1Pk'
    __MW_PANEL = 'http://127.0.0.1:7200'

    def __init__(self, mw_panel=None, mw_key=None):
        if False:
            i = 10
            return i + 15
        if mw_panel:
            self.__MW_PANEL = mw_panel
            self.__MW_KEY = mw_key

    def __get_md5(self, s):
        if False:
            return 10
        m = hashlib.md5()
        m.update(s.encode('utf-8'))
        return m.hexdigest()

    def __get_key_data(self):
        if False:
            for i in range(10):
                print('nop')
        now_time = int(time.time())
        ready_data = {'request_token': self.__get_md5(str(now_time) + '' + self.__get_md5(self.__MW_KEY)), 'request_time': now_time}
        return ready_data

    def __http_post_cookie(self, url, p_data, timeout=1800):
        if False:
            while True:
                i = 10
        cookie_file = '/tmp/' + self.__get_md5(self.__MW_PANEL) + '.cookie'
        if sys.version_info[0] == 2:
            import urllib
            import urllib2
            import ssl
            import cookielib
            cookie_obj = cookielib.MozillaCookieJar(cookie_file)
            if os.path.exists(cookie_file):
                cookie_obj.load(cookie_file, ignore_discard=True, ignore_expires=True)
            ssl._create_default_https_context = ssl._create_unverified_context
            data = urllib.urlencode(p_data)
            req = urllib2.Request(url, data)
            opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie_obj))
            response = opener.open(req, timeout=timeout)
            cookie_obj.save(ignore_discard=True, ignore_expires=True)
            return response.read()
        else:
            import urllib.request
            import ssl
            import http.cookiejar
            cookie_obj = http.cookiejar.MozillaCookieJar(cookie_file)
            if os.path.exists(cookie_file):
                cookie_obj.load(cookie_file, ignore_discard=True, ignore_expires=True)
            handler = urllib.request.HTTPCookieProcessor(cookie_obj)
            data = urllib.parse.urlencode(p_data).encode('utf-8')
            req = urllib.request.Request(url, data)
            opener = urllib.request.build_opener(handler)
            response = opener.open(req, timeout=timeout)
            cookie_obj.save(ignore_discard=True, ignore_expires=True)
            result = response.read()
            if type(result) == bytes:
                result = result.decode('utf-8')
            return result

    def getLogs(self):
        if False:
            i = 10
            return i + 15
        url = self.__MW_PANEL + '/api/logs/get_log_list'
        post_data = self.__get_key_data()
        post_data['limit'] = 10
        post_data['p'] = '1'
        result = self.__http_post_cookie(url, post_data)
        return json.loads(result)
if __name__ == '__main__':
    api = mwApi()
    rdata = api.getLogs()
    print(rdata)