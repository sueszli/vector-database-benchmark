"""
RxPY example running a Tornado server doing search queries against
Wikipedia to populate the autocomplete dropdown in the web UI. Start
using `python autocomplete.py` and navigate your web browser to
http://localhost:8080

Uses the RxPY AsyncIOScheduler (Python 3.4 is required)
"""
import asyncio
import os
from asyncio import Future
from typing import Dict, Union
from tornado.escape import json_decode
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httputil import url_concat
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.web import Application, RequestHandler, StaticFileHandler, url
from tornado.websocket import WebSocketHandler
from reactivex import operators as ops
from reactivex.scheduler.eventloop import AsyncIOScheduler
from reactivex.subject import Subject

def search_wikipedia(term: str) -> Future[HTTPResponse]:
    if False:
        while True:
            i = 10
    'Search Wikipedia for a given term'
    url = 'http://en.wikipedia.org/w/api.php'
    params = {'action': 'opensearch', 'search': term, 'format': 'json'}
    user_agent = 'RxPY/1.0 (https://github.com/dbrattli/RxPY; dag@brattli.net) Tornado/4.0.1'
    url = url_concat(url, params)
    http_client = AsyncHTTPClient()
    return http_client.fetch(url, method='GET', user_agent=user_agent)

class WSHandler(WebSocketHandler):

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        scheduler = AsyncIOScheduler(asyncio.get_event_loop())
        print('WebSocket opened')
        self.subject: Subject[Dict[str, str]] = Subject()
        searcher = self.subject.pipe(ops.map(lambda x: x['term']), ops.filter(lambda text: len(text) > 2), ops.debounce(0.75), ops.distinct_until_changed(), ops.flat_map_latest(search_wikipedia))

        def send_response(x: HTTPResponse) -> None:
            if False:
                return 10
            self.write_message(x.body)

        def on_error(ex: Exception):
            if False:
                while True:
                    i = 10
            print(ex)
        searcher.subscribe(on_next=send_response, on_error=on_error, scheduler=scheduler)

    def on_message(self, message: Union[bytes, str]):
        if False:
            return 10
        obj = json_decode(message)
        self.subject.on_next(obj)

    def on_close(self):
        if False:
            for i in range(10):
                print('nop')
        print('WebSocket closed')

class MainHandler(RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        self.render('index.html')

def main():
    if False:
        return 10
    AsyncIOMainLoop().make_current()
    port = os.environ.get('PORT', 8080)
    app = Application([url('/', MainHandler), ('/ws', WSHandler), ('/static/(.*)', StaticFileHandler, {'path': '.'})])
    print('Starting server at port: %s' % port)
    app.listen(port)
    asyncio.get_event_loop().run_forever()
if __name__ == '__main__':
    main()