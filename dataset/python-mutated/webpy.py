"""A web.py application powered by gevent"""
from __future__ import print_function
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
import time
import web
urls = ('/', 'index', '/long', 'long_polling')

class index(object):

    def GET(self):
        if False:
            while True:
                i = 10
        return '<html>Hello, world!<br><a href="/long">/long</a></html>'

class long_polling(object):

    def GET(self):
        if False:
            for i in range(10):
                print('nop')
        print('GET /long')
        time.sleep(10)
        return 'Hello, 10 seconds later'
if __name__ == '__main__':
    application = web.application(urls, globals()).wsgifunc()
    print('Serving on 8088...')
    WSGIServer(('', 8088), application).serve_forever()