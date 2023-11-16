"""
The automcomplete example rewritten for bottle / gevent.
- Requires besides bottle and gevent also the geventwebsocket pip package
- Instead of a future we create the inner stream for flat_map_latest manually
"""
import json
import gevent
import requests
from bottle import Bottle, abort, request
from geventwebsocket import WebSocketError
from geventwebsocket.handler import WebSocketHandler
from reactivex.scheduler.eventloop import GEventScheduler
from reactivex.subject import Subject

class WikiFinder:
    tmpl = 'http://en.wikipedia.org/w/api.php'
    tmpl += '?action=opensearch&search=%s&format=json'

    def __init__(self, term):
        if False:
            i = 10
            return i + 15
        self.res = r = gevent.event.AsyncResult()
        gevent.spawn(lambda : requests.get(self.tmpl % term).text).link(r)

    def subscribe(self, on_next, on_err, on_compl):
        if False:
            while True:
                i = 10
        try:
            self.res.get()
            on_next(self.res.value)
        except Exception as ex:
            on_err(ex.args)
        on_compl()
(app, PORT) = (Bottle(), 8081)
scheduler = GEventScheduler(gevent)

@app.route('/ws')
def handle_websocket():
    if False:
        print('Hello World!')
    wsock = request.environ.get('wsgi.websocket')
    if not wsock:
        abort(400, 'Expected WebSocket request.')
    stream = Subject()
    query = stream.map(lambda x: x['term']).filter(lambda text: len(text) > 2).debounce(0.75, scheduler=scheduler).distinct_until_changed()
    searcher = query.flat_map_latest(lambda term: WikiFinder(term))

    def send_response(x):
        if False:
            i = 10
            return i + 15
        wsock.on_next(x)

    def on_error(ex):
        if False:
            print('Hello World!')
        print(ex)
    searcher.subscribe(send_response, on_error)
    while True:
        try:
            message = wsock.receive()
            obj = json.loads(message)
            stream.on_next(obj)
        except WebSocketError:
            break

@app.route('/static/autocomplete.js')
def get_js():
    if False:
        while True:
            i = 10
    return open('autocomplete.js').read().replace('8080', str(PORT))

@app.route('/')
def get_index():
    if False:
        for i in range(10):
            print('nop')
    return open('index.html').read()
if __name__ == '__main__':
    h = ('0.0.0.0', PORT)
    server = gevent.pywsgi.WSGIServer(h, app, handler_class=WebSocketHandler)
    server.serve_forever()